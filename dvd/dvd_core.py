import copy
import json
from typing import Annotated as A
from typing import Literal as L

import dvd.config as config
from dvd.build_database import (clip_search_tool, frame_inspect_tool,
                                global_browse_tool, init_single_video_db)
from dvd.func_call_shema import as_json_schema
from dvd.func_call_shema import doc as D
from dvd.utils import call_openai_model_with_tools

TOPK = 16

class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


def finish(answer: A[str, D("Answer to the user's question.")]) -> None:
    """Call this function after confirming the answer of the user's question, and finish the conversation."""
    raise StopException(answer)


class DVDCoreAgent:
    def __init__(self, video_db_path, video_caption_path, max_iterations):
        self.tools = [frame_inspect_tool, clip_search_tool, global_browse_tool, finish]
        if config.LITE_MODE:
            self.tools.remove(frame_inspect_tool)
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        self.video_db = init_single_video_db(video_caption_path, video_db_path, config.AOAI_EMBEDDING_LARGE_DIM)
        self.max_iterations = max_iterations
        self.messages = self._construct_messages()

    def _construct_messages(self):
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the THINK → ACT → OBSERVE loop:
  • THOUGHT Reason step-by-step about which function to call next.
  • ACTION   Call exactly one function that moves you closer to the final answer.
  • OBSERVATION Summarize the function's output.
You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls.
Only pass arguments that come verbatim from the user or from earlier function outputs—never invent them. Continue the loop until the user's query is fully resolved, then end your turn with the final answer. If you are uncertain about code structure or video content, use the available tools to inspect rather than guessing. Plan carefully before each call and reflect on every result. Do not rely solely on blind function calls, as this degrades reasoning quality. Timestamps may be formatted as 'HH:MM:SS' or 'MM:SS'."""
            },
            {
                "role": "user",
                "content": """Carefully read the timestamps and narration in the following script, paying attention to the causal order of events, object details and movements, and people's actions and poses.

Here are tools you can use to reveal your reasoning process whenever the provided information is insufficient.

• To get a global information about events and main subjects in the video, use `global_browse_tool`.
• To search without a specific timestamp, use `clip_search_tool`.
• If the retrieved material lacks precise, question-relevant detail (e.g., an unknown name), call `frame_inspect_tool` with a list of time ranges (list[tuple[HH:MM:SS, HH:MM:SS]]).
• Whenever you are uncertain of an answer after searching, inspect frames in the relevant intervals with `frame_inspect_tool`.
• After locating an answer in the script, always make a **CONFIRM** with `frame_inspect_tool` query.


You can first use `global_browse_tool` to a global information about this video, then invoke multiple times of these tools to prgressively find the answer.

Based on your observations and tool outputs, provide a concise answer that directly addresses the question. \n

Total video length: VIDEO_LENGTH seconds.

Question: QUESTION_PLACEHOLDER"""
            },
        ]
        video_length = self.video_db.get_additional_data()['video_length']
        messages[-1]['content'] = messages[-1]['content'].replace("VIDEO_LENGTH", str(video_length))
        return messages

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        msgs.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": name,
                "content": content,
            }
        )

    def _exec_tool(self, tool_call, msgs):
        name = tool_call["function"]["name"]
        if name not in self.name_to_function_map:
            self._append_tool_msg(tool_call["id"], name, f"Invalid function name: {name!r}", msgs)
            return

        # Parse arguments
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as exc:
            raise StopException(f"Error decoding arguments: {exc!s}")

        if "database" in args:
            args["database"] = self.video_db
        
        if "topk" in args:
            if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
                args["topk"] = config.OVERWRITE_CLIP_SEARCH_TOPK

        # Call the tool
        try:
            print(f"Calling function `{name}` with args: {args}")
            result = self.name_to_function_map[name](**args)
            self._append_tool_msg(tool_call["id"], name, result, msgs)
        except StopException as exc:  # graceful stop
            print(f"Finish task with message: '{exc!s}'")
            raise

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, question) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.
        """
        msgs = copy.deepcopy(self.messages)
        msgs[-1]["content"] = msgs[-1]["content"].replace("QUESTION_PLACEHOLDER", question)

        for i in range(self.max_iterations):
            # Force a final `finish` on the last iteration to avoid hanging
            if i == self.max_iterations - 1:
                msgs.append(
                    {
                        "role": "user",
                        "content": "Please call the `finish` function to finish the task.",
                    }
                )

            response = call_openai_model_with_tools(
                msgs,
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                tools=self.function_schemas,
                temperature=0.0,
                api_key=config.OPENAI_API_KEY,
            )
            if response is None:
                return None

            response.setdefault("role", "assistant")
            msgs.append(response)

            # Execute any requested tool calls
            try:
                for tool_call in response.get("tool_calls", []):
                    self._exec_tool(tool_call, msgs)
            except StopException:
                return msgs

        return msgs

    # ------------------------------------------------------------------ #
    # Streaming (generator) loop
    # ------------------------------------------------------------------ #
    def stream_run(self, question):
        """
        A generator version of `run`.  
        Yields:
            dict: every assistant / tool message produced during reasoning.
        """
        msgs = copy.deepcopy(self.messages)
        msgs[-1]["content"] = msgs[-1]["content"].replace("QUESTION_PLACEHOLDER", question)

        for i in range(self.max_iterations):
            # Force a final `finish` on the last iteration
            if i == self.max_iterations - 1:
                final_usr_msg = {
                    "role": "user",
                    "content": "Please call the `finish` function to finish the task.",
                }
                msgs.append(final_usr_msg)
                # Don't yield user messages to the UI

            response = call_openai_model_with_tools(
                msgs,
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                tools=self.function_schemas,
                temperature=0.0,
                api_key=config.OPENAI_API_KEY,
            )
            if response is None:
                return

            response.setdefault("role", "assistant")
            msgs.append(response)
            yield response  # ← stream assistant reply

            # Execute any requested tool calls
            try:
                for tool_call in response.get("tool_calls", []):
                    # Yield a formatted message about the tool being called
                    tool_name = tool_call.get("function", {}).get("name", "unknown")
                    tool_args = tool_call.get("function", {}).get("arguments", "{}")
                    yield {
                        "role": "tool_call",
                        "name": tool_name,
                        "arguments": tool_args
                    }
                    
                    self._exec_tool(tool_call, msgs)
                    # Only yield the tool result message
                    if msgs[-1].get("role") == "tool":
                        yield msgs[-1]  # ← stream tool observation
            except StopException:
                return


def single_run_wrapper(info) -> dict:
    qid, video_db_path, video_caption_path, question = info
    agent = DVDCoreAgent(video_db_path, video_caption_path, question)
    msgs = agent.run()
    return {qid: msgs}


def main():
    video_caption_path = "./video_database/i2qSxMVeVLI/captions/captions.json"
    video_db_path = "./video_database/i2qSxMVeVLI/database.json"
    question = """What color are the lights on the coaches' chairs when they turn around?
(A) Red
(B) White
(C) Black
(D) Yellow"""
    
    agent = DVDCoreAgent(video_db_path, video_caption_path, max_iterations=15)
    messages = agent.run(question)

    if messages:
        for m in messages:
            import pdb; pdb.set_trace()
            if m.get('role') == 'assistant':
                print(m)


if __name__ == "__main__":
    main()
