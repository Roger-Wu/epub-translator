import unittest
from pathlib import Path
from xml.etree.ElementTree import Element, fromstring, tostring

from epub_translator.segment import search_text_segments
from epub_translator.translation.translator import _apply_submit, _normalize_submits, _resolve_target_path
from epub_translator.utils import normalize_whitespace
from epub_translator.xml_translator.stream_mapper import InlineSegmentMapping
from epub_translator.xml_translator.submitter import SubmitKind
from scripts.translate_epub import _parse_submit_modes


def parse_xml(xml_str: str) -> Element:
    return fromstring(xml_str.strip())


def element_to_string(element: Element) -> str:
    return tostring(element, encoding="unicode", method="html").rstrip()


class TestMultiOutputHelpers(unittest.TestCase):
    def test_normalize_submits_keeps_order_and_deduplicates(self):
        submits = _normalize_submits(
            [
                SubmitKind.REPLACE,
                SubmitKind.APPEND_BLOCK,
                SubmitKind.REPLACE,
            ]
        )

        self.assertEqual(submits, (SubmitKind.REPLACE, SubmitKind.APPEND_BLOCK))

    def test_resolve_target_path_for_multiple_outputs(self):
        target_path = Path("translated.epub")

        resolved = _resolve_target_path(target_path, SubmitKind.APPEND_BLOCK, multiple=True)

        self.assertEqual(resolved, Path("translated.append_block.epub"))

    def test_parse_submit_modes_defaults_to_replace_and_append_block(self):
        submit_modes = _parse_submit_modes(None)

        self.assertEqual(submit_modes, [SubmitKind.REPLACE, SubmitKind.APPEND_BLOCK])

    def test_apply_submit_can_remap_mappings_to_another_tree(self):
        source_root = parse_xml(
            """
            <body>
                <p id="p1">hello world</p>
            </body>
            """
        )
        source_paragraph = source_root.find("./p")
        assert source_paragraph is not None

        target_root = parse_xml(
            """
            <body>
                <p id="p1">hello world</p>
            </body>
            """
        )

        translated_xml = parse_xml("<p>你好世界</p>")
        translated_segments = list(search_text_segments(translated_xml))
        mappings: list[InlineSegmentMapping] = [(source_paragraph, translated_segments)]

        result = _apply_submit(
            source_element=source_root,
            target_element=target_root,
            action=SubmitKind.APPEND_BLOCK,
            mappings=mappings,
        )

        expected = """
        <body>
            <p id="p1">hello world</p><p>你好世界</p>
        </body>
        """
        self.assertIs(result, target_root)
        self.assertEqual(
            normalize_whitespace(element_to_string(result)).strip(),
            normalize_whitespace(expected).strip(),
        )
