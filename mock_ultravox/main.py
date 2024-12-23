
class VoiceOutput:
    def __init__(self, text: str):
        self.text = text

class LocalInference:
    """
    Mock version of LocalInference with a simple infer() method that
    just echoes back the last user message.
    """
    def __init__(
        self,
        model,
        processor,
        tokenizer,
        device: str,
        dtype,
        conversation_mode: bool = False,
    ):
        # For a real implementation, you'd store these parameters.
        pass

    def infer(self, sample, max_tokens=None, temperature=None):
        """
        Echoes back the last user message in sample.messages.
        """
        # Assume sample has a .messages list of dicts: 
        # [{"role":"user","content":"some message"}, ...].
        # We'll just retrieve the last message content.
        last_message = sample.messages[-1]["content"]
        # Return a VoiceOutput object that holds that text.
        return VoiceOutput(f"Echo: {last_message}")
