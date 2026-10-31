# Supported operations

Generated from `stc.ops.supported_ops()`. Anything else stops tracing with `Unsupported` naming the call.

## Functions

`torch.abs`, `torch.add`, `torch.amax`, `torch.amin`, `torch.arange`, `torch.bmm`, `torch.cat`, `torch.concat`, `torch.cos`, `torch.div`, `torch.empty`, `torch.eq`, `torch.exp`, `torch.flatten`, `torch.full`, `torch.full_like`, `torch.gt`, `torch.log`, `torch.logical_and`, `torch.logical_not`, `torch.logical_or`, `torch.logsumexp`, `torch.lt`, `torch.matmul`, `torch.maximum`, `torch.mean`, `torch.minimum`, `torch.mm`, `torch.mul`, `torch.ne`, `torch.neg`, `torch.nn.functional.cross_entropy`, `torch.nn.functional.dropout`, `torch.nn.functional.embedding`, `torch.nn.functional.gelu`, `torch.nn.functional.layer_norm`, `torch.nn.functional.linear`, `torch.nn.functional.log_softmax`, `torch.nn.functional.relu`, `torch.nn.functional.rms_norm`, `torch.nn.functional.scaled_dot_product_attention`, `torch.nn.functional.sigmoid`, `torch.nn.functional.silu`, `torch.nn.functional.softmax`, `torch.nn.functional.tanh`, `torch.ones`, `torch.ones_like`, `torch.outer`, `torch.permute`, `torch.pow`, `torch.relu`, `torch.reshape`, `torch.rsqrt`, `torch.sigmoid`, `torch.sin`, `torch.softmax`, `torch.split`, `torch.sqrt`, `torch.squeeze`, `torch.stack`, `torch.sub`, `torch.sum`, `torch.tanh`, `torch.tensor`, `torch.transpose`, `torch.tril`, `torch.triu`, `torch.unsqueeze`, `torch.var`, `torch.where`, `torch.zeros`, `torch.zeros_like`

## Tensor methods and operators

`.abs`, `.add`, `.all`, `.amax`, `.amin`, `.any`, `.argmax`, `.bfloat16`, `.bool`, `.chunk`, `.clamp`, `.clone`, `.contiguous`, `.cos`, `.cumsum`, `.detach`, `.div`, `.double`, `.eq`, `.exp`, `.expand`, `.expand_as`, `.flatten`, `.float`, `.floor_divide`, `.ge`, `.gt`, `.half`, `.int`, `.isnan`, `.le`, `.log`, `.log_softmax`, `.logical_and`, `.logical_not`, `.logical_or`, `.logsumexp`, `.long`, `.lt`, `.masked_fill`, `.matmul`, `.maximum`, `.mean`, `.minimum`, `.mul`, `.ne`, `.neg`, `.permute`, `.pow`, `.reciprocal`, `.relu`, `.remainder`, `.reshape`, `.rsqrt`, `.sigmoid`, `.sin`, `.size`, `.softmax`, `.split`, `.sqrt`, `.square`, `.squeeze`, `.std`, `.sub`, `.sum`, `.t`, `.tanh`, `.to`, `.transpose`, `.tril`, `.triu`, `.type_as`, `.unsqueeze`, `.var`, `.view`

Also: arithmetic and comparison operators on tensors (`+ - * / ** @ < <= > >= == != & | ~`), unary minus, indexing with ints, slices (positive step), `None` and `...`, the attributes `.shape .ndim .dtype .device .T .mT`, `.size()`, `.dim()`, `.numel()`, and `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.RMSNorm`, `nn.Dropout`, `nn.GELU`/`nn.SiLU`/`nn.ReLU` modules (they call the functions above).

Not supported: indexing with tensors or lists, `.item()`, data-dependent control flow, in-place operations, random tensors created inside the program.
