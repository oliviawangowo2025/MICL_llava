from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model_was
from llava.utils import disable_torch_init


class llava_bot:
    def __init__(self, model_path):
        self.model_path = model_path
        """self.args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(self.model_path),
            "query": None,
            "conv_mode": None,
            "image_file": None,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
        """
        disable_torch_init()
        self.model_name = get_model_name_from_path(model_path)
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        model_path, None, self.model_name, load_4bit=True
        ) # modify: load 4 bit

    def ask_llava(self, prompt, image_file):

        response = eval_model_was(prompt, image_file, self.model, self.model_name, self.image_processor, self.tokenizer)

        #response = eval_model(args)
        print("prompt: ", prompt)
        print("response: ", response)
        return response


