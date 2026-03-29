from collections.abc import Callable, Generator, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum, auto
from importlib.metadata import version as get_package_version
from os import PathLike
from pathlib import Path
from xml.etree.ElementTree import Element

from ..epub import (
    MetadataContext,
    TocContext,
    Zip,
    read_metadata,
    read_toc,
    search_spine_paths,
    write_metadata,
    write_toc,
)
from ..llm import LLM
from ..xml import XMLLikeNode, clone_element, deduplicate_ids_in_element, find_first, iter_with_stack
from ..xml_translator import FillFailedEvent, SubmitKind, TranslationTask, XMLTranslator
from .epub_transcode import decode_metadata, decode_toc_list, encode_metadata, encode_toc_list
from .punctuation import unwrap_french_quotes
from .xml_interrupter import XMLInterrupter


class _ElementType(Enum):
    TOC = auto()
    METADATA = auto()
    CHAPTER = auto()


@dataclass
class _ElementContext:
    output_zip: Zip
    submit: SubmitKind
    action: SubmitKind
    element_type: _ElementType
    target_element: Element | None = None
    chapter_data: tuple[Path, XMLLikeNode] | None = None
    toc_context: TocContext | None = None
    metadata_context: MetadataContext | None = None


def translate(
    source_path: PathLike | str,
    target_path: PathLike | str,
    target_language: str,
    submit: SubmitKind | Iterable[SubmitKind],
    user_prompt: str | None = None,
    max_retries: int = 5,
    max_group_tokens: int = 2600,
    concurrency: int = 1,
    llm: LLM | None = None,
    translation_llm: LLM | None = None,
    fill_llm: LLM | None = None,
    on_progress: Callable[[float], None] | None = None,
    on_fill_failed: Callable[[FillFailedEvent], None] | None = None,
) -> None:
    submits = _normalize_submits(submit)
    translation_llm = translation_llm or llm
    fill_llm = fill_llm or llm
    if translation_llm is None:
        raise ValueError("Either translation_llm or llm must be provided")
    if fill_llm is None:
        raise ValueError("Either fill_llm or llm must be provided")

    translator = XMLTranslator(
        translation_llm=translation_llm,
        fill_llm=fill_llm,
        target_language=target_language,
        user_prompt=user_prompt,
        ignore_translated_error=False,
        max_retries=max_retries,
        max_fill_displaying_errors=10,
        max_group_score=max_group_tokens,
        cache_seed_content=f"{_get_version()}:{target_language}",
    )
    source_path = Path(source_path).resolve()
    target_path = Path(target_path).resolve()

    with ExitStack() as stack:
        outputs = [
            _OutputBook(
                submit=current_submit,
                target_path=_resolve_target_path(target_path, current_submit, multiple=len(submits) > 1),
                zip=stack.enter_context(
                    Zip(
                        source_path=source_path,
                        target_path=_resolve_target_path(target_path, current_submit, multiple=len(submits) > 1),
                    )
                ),
            )
            for current_submit in submits
        ]

        for output in outputs:
            # mimetype should be the first file in the EPUB ZIP
            output.zip.migrate(Path("mimetype"))

        source_zip = outputs[0].zip
        total_chapters = sum(1 for _, _ in search_spine_paths(source_zip))
        toc_list, _ = read_toc(source_zip)
        metadata_fields, _ = read_metadata(source_zip)

        toc_has_items = len(toc_list) > 0
        metadata_has_items = len(metadata_fields) > 0
        total_items = (1 if toc_has_items else 0) + (1 if metadata_has_items else 0) + total_chapters

        if total_items == 0:
            return

        interrupter = XMLInterrupter()
        toc_weight = 0.05 if toc_has_items else 0
        metadata_weight = 0.05 if metadata_has_items else 0
        chapters_weight = 1.0 - toc_weight - metadata_weight
        progress_per_chapter = chapters_weight / total_chapters if total_chapters > 0 else 0
        current_progress = 0.0

        for source_element, mappings, task in translator.translate_mapped_elements(
            concurrency=concurrency,
            interrupt_source_text_segments=interrupter.interrupt_source_text_segments,
            interrupt_translated_text_segments=interrupter.interrupt_translated_text_segments,
            interrupt_block_element=interrupter.interrupt_block_element,
            on_fill_failed=on_fill_failed,
            tasks=_generate_tasks_from_book(
                source_zip=source_zip,
                outputs=outputs,
                toc_list=toc_list,
                metadata_fields=metadata_fields,
            ),
        ):
            for context in task.payload:
                translated_elem = _apply_submit(
                    source_element=source_element,
                    target_element=context.target_element,
                    action=context.action,
                    mappings=mappings,
                )
                _write_translated_element(context, translated_elem)

            if task.payload and task.payload[0].element_type == _ElementType.TOC:
                current_progress += toc_weight
            elif task.payload and task.payload[0].element_type == _ElementType.METADATA:
                current_progress += metadata_weight
            elif task.payload and task.payload[0].element_type == _ElementType.CHAPTER:
                current_progress += progress_per_chapter

            if on_progress:
                on_progress(current_progress)


def _generate_tasks_from_book(
    source_zip: Zip,
    outputs: list["_OutputBook"],
    toc_list: list,
    metadata_fields: list,
) -> Generator[TranslationTask[list[_ElementContext]], None, None]:
    if toc_list:
        contexts = []
        for output in outputs:
            _, toc_context = read_toc(output.zip)
            contexts.append(
                _ElementContext(
                    output_zip=output.zip,
                    submit=output.submit,
                    action=_to_head_submit(output.submit),
                    element_type=_ElementType.TOC,
                    toc_context=toc_context,
                )
            )
        yield TranslationTask(
            element=encode_toc_list(toc_list),
            action=contexts[0].action,
            payload=contexts,
        )

    if metadata_fields:
        contexts = []
        for output in outputs:
            _, metadata_context = read_metadata(output.zip)
            contexts.append(
                _ElementContext(
                    output_zip=output.zip,
                    submit=output.submit,
                    action=_to_head_submit(output.submit),
                    element_type=_ElementType.METADATA,
                    metadata_context=metadata_context,
                )
            )
        yield TranslationTask(
            element=encode_metadata(metadata_fields),
            action=contexts[0].action,
            payload=contexts,
        )

    for chapter_path, media_type in search_spine_paths(source_zip):
        with source_zip.read(chapter_path) as chapter_file:
            source_xml = XMLLikeNode(
                file=chapter_file,
                is_html_like=(media_type == "text/html"),
            )
        source_body_element = find_first(source_xml.element, "body")
        if source_body_element is not None:
            contexts = []
            for output in outputs:
                with output.zip.read(chapter_path) as chapter_file:
                    xml = XMLLikeNode(
                        file=chapter_file,
                        is_html_like=(media_type == "text/html"),
                    )
                body_element = find_first(xml.element, "body")
                if body_element is None:
                    continue
                contexts.append(
                    _ElementContext(
                        output_zip=output.zip,
                        submit=output.submit,
                        action=output.submit,
                        element_type=_ElementType.CHAPTER,
                        target_element=body_element,
                        chapter_data=(chapter_path, xml),
                    )
                )
            yield TranslationTask(
                element=source_body_element,
                action=contexts[0].action,
                payload=contexts,
            )


@dataclass
class _OutputBook:
    submit: SubmitKind
    target_path: Path
    zip: Zip


def _normalize_submits(submit: SubmitKind | Iterable[SubmitKind]) -> tuple[SubmitKind, ...]:
    if isinstance(submit, SubmitKind):
        return (submit,)

    normalized: list[SubmitKind] = []
    for item in submit:
        if not isinstance(item, SubmitKind):
            raise TypeError("submit must be a SubmitKind or an iterable of SubmitKind values")
        if item not in normalized:
            normalized.append(item)

    if not normalized:
        raise ValueError("submit iterable must contain at least one SubmitKind")

    return tuple(normalized)


def _resolve_target_path(target_path: Path, submit: SubmitKind, multiple: bool) -> Path:
    if not multiple:
        return target_path

    return target_path.with_name(f"{target_path.stem}.{submit.name.lower()}{target_path.suffix}")


def _to_head_submit(submit: SubmitKind) -> SubmitKind:
    if submit == SubmitKind.APPEND_BLOCK:
        return SubmitKind.APPEND_TEXT
    return submit


def _apply_submit(
    source_element: Element,
    target_element: Element | None,
    action: SubmitKind,
    mappings,
) -> Element:
    if target_element is None:
        cloned_element, element_map = _clone_with_element_map(source_element)
        remapped_mappings = [(element_map[id(element)], text_segments) for element, text_segments in mappings]
        return _submit_element(cloned_element, action, remapped_mappings)

    if target_element is source_element:
        return _submit_element(target_element, action, mappings)

    element_map = _build_element_map(source_element, target_element)
    remapped_mappings = [(element_map[id(element)], text_segments) for element, text_segments in mappings]
    return _submit_element(target_element, action, remapped_mappings)


def _submit_element(element: Element, action: SubmitKind, mappings):
    from ..xml_translator.submitter import submit as submit_element

    return submit_element(element=element, action=action, mappings=mappings)


def _clone_with_element_map(element: Element) -> tuple[Element, dict[int, Element]]:
    cloned_element = clone_element(element)
    return cloned_element, _build_element_map(element, cloned_element)


def _build_element_map(source_element: Element, target_element: Element) -> dict[int, Element]:
    element_map: dict[int, Element] = {}
    source_iter = iter_with_stack(source_element)
    target_iter = iter_with_stack(target_element)

    for (_, source), (_, target) in zip(source_iter, target_iter, strict=True):
        if source.tag != target.tag or len(source) != len(target):
            raise ValueError("Source and target elements do not share the same structure")
        element_map[id(source)] = target

    return element_map


def _write_translated_element(context: _ElementContext, translated_elem: Element) -> None:
    if context.element_type == _ElementType.TOC:
        translated_elem = unwrap_french_quotes(translated_elem)
        decoded_toc = decode_toc_list(translated_elem)
        if context.toc_context is not None:
            write_toc(context.output_zip, decoded_toc, context.toc_context)
        return

    if context.element_type == _ElementType.METADATA:
        translated_elem = unwrap_french_quotes(translated_elem)
        decoded_metadata = decode_metadata(translated_elem)
        if context.metadata_context is not None:
            write_metadata(context.output_zip, decoded_metadata, context.metadata_context)
        return

    if context.element_type == _ElementType.CHAPTER and context.chapter_data is not None:
        chapter_path, xml = context.chapter_data
        deduplicate_ids_in_element(xml.element)
        with context.output_zip.replace(chapter_path) as target_file:
            xml.save(target_file)


def _get_version() -> str:
    try:
        return get_package_version("epub-translator")
    except Exception:
        return "development"
