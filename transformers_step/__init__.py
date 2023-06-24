__version__ = "2.1.0"

try:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass

import logging

logger = logging.getLogger(__name__)

# Files and general utilities
from .file_utils import (TRANSFORMERS_CACHE, PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME,
                         is_tf_available, is_torch_available)

from .data import (
                   InputExample, InputFeatures, DataProcessor,
                   glue_output_modes, glue_convert_examples_to_features,
                   glue_processors, glue_tasks_num_labels)

from .configuration_utils import PretrainedConfig

from .configuration_bert import BertConfig, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

from .encoder import TransformerInterEncoder

# Modeling
if is_torch_available():
    from .modeling_utils import (PreTrainedModel, prune_layer, Conv1D)

    from .modeling_bert import pointer_eval

    from .optimization import (AdamW, ConstantLRSchedule, WarmupConstantSchedule, WarmupCosineSchedule,
                               WarmupCosineWithHardRestartsSchedule, WarmupLinearSchedule)


