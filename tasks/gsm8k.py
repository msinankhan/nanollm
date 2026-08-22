import re
from tasks.common import Task, load_hub_dataset

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    match= GSM_RE.search(completion)

    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",","")
        return match_str
    return None


class GSM8K(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_hub_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row['question']
        answer = row['answer']

        assistant_message_parts = []

        parts = re.split(r'(<<[^>]+>>)', answer)

        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                inner = part[2:-2]

                if '=' in inner:
                    expr, result= inner.rsplit('=',1)

                else:
                    expr, result = inner, ""

                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append({"type":"python_output", "text": result})

            else:
                assistant_message_parts.append({"type": "text", "text": part})
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_message_parts},

        ]

        conversation = {
            "messages": messages,
        }

        return conversation

    def evaluate(self, conversation, assistant_response):

        assert isinstance(assistant_response,str), "Assuming simple string responses for now."

        assistant_message = conversation['messages'][-1]

        assert assistant_message['role'] =='assistant', "Last message must be from the assistant"

        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts."

        last_text_part = assistant_message['content'][-1]['text']

        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)

        is_correct = int(pred_num == ref_num)

        return is_correct

    def reward(self,conversation, assistant_response):

        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)

        return is_correct_float
        