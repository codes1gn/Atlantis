from transformers import BertForSequenceClassification
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
from transformers import AutoModelForCausalLM
from transformers.models.resnet import (
    ResNetConfig,
    ResNetForImageClassification,
    ResNetModel,
)
from transformers import ViTConfig, ViTForImageClassification, ViTModel
from transformers import (
    ResNetConfig,
    ResNetForImageClassification,
)

resnet = ResNetForImageClassification(ResNetConfig(num_labels=10))

transformers_bert = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=1000, output_attentions=False, output_hidden_states=False)
# transformers_bert = ''

transformers_resnet = ResNetForImageClassification(
    ResNetConfig(num_labels=(int)(10)))

transformers_gpt_neo = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-2.7B", low_cpu_mem_usage=True)
# transformers_gpt_neo = ''

# transformers_deit_b = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-base-distilled-patch16-224')
transformers_deit_b = ''
