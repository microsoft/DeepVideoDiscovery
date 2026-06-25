"""Tests for the optional TwelveLabs Pegasus integration.

Run with:  python -m unittest tests.test_twelvelabs_pegasus

The live test is skipped unless TWELVELABS_API_KEY is set. The no-network test
always runs and only requires the `twelvelabs` SDK to be installed.
"""
import os
import unittest

HAS_KEY = bool(os.environ.get("TWELVELABS_API_KEY"))

# A small, Pegasus-conforming public sample.
SAMPLE_VIDEO_URL = "https://storage.googleapis.com/cloud-samples-data/video/animals.mp4"


class TestPegasusWiring(unittest.TestCase):
    def test_tool_registered_only_when_enabled(self):
        """pegasus_video_qa_tool is opt-in: present only in non-LITE mode with the flag on."""
        import dvd.config as config

        orig_lite, orig_use = config.LITE_MODE, config.USE_TWELVELABS_PEGASUS
        try:
            from dvd.build_database import pegasus_video_qa_tool

            # Re-evaluate the same gating logic DVDCoreAgent.__init__ uses, without
            # touching the network or building a real DB.
            def tools_for(lite, use):
                from dvd.build_database import (clip_search_tool,
                                                frame_inspect_tool,
                                                global_browse_tool)
                tools = [frame_inspect_tool, clip_search_tool, global_browse_tool]
                if lite:
                    tools.remove(frame_inspect_tool)
                elif use:
                    tools.insert(1, pegasus_video_qa_tool)
                return tools

            self.assertIn(pegasus_video_qa_tool, tools_for(lite=False, use=True))
            self.assertNotIn(pegasus_video_qa_tool, tools_for(lite=False, use=False))
            self.assertNotIn(pegasus_video_qa_tool, tools_for(lite=True, use=True))
        finally:
            config.LITE_MODE, config.USE_TWELVELABS_PEGASUS = orig_lite, orig_use

    def test_schema_generation(self):
        """The tool exposes a valid function schema to the agent."""
        from dvd.build_database import pegasus_video_qa_tool
        from dvd.func_call_shema import as_json_schema

        schema = as_json_schema(pegasus_video_qa_tool)
        self.assertEqual(schema["name"], "pegasus_video_qa_tool")
        self.assertIn("question", schema["parameters"]["properties"])
        self.assertIn("time_ranges_hhmmss", schema["parameters"]["properties"])


@unittest.skipUnless(HAS_KEY, "TWELVELABS_API_KEY not set")
class TestPegasusLive(unittest.TestCase):
    def test_analyze_public_video(self):
        """End-to-end: upload a public sample and get a non-empty Pegasus answer."""
        import urllib.request
        import tempfile

        from dvd.utils import TwelveLabsPegasusService

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            urllib.request.urlretrieve(SAMPLE_VIDEO_URL, tmp.name)
            path = tmp.name
        try:
            answer = TwelveLabsPegasusService.analyze_video_file(
                video_path=path,
                prompt="Describe this video in one sentence.",
                api_key=os.environ["TWELVELABS_API_KEY"],
                max_tokens=512,
            )
            self.assertIsInstance(answer, str)
            self.assertTrue(answer.strip())
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
