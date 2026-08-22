import torch 
import torch.nn as nn

from nanollm.commons import COMPUTE_DTYPE

EPS=1e-12


@torch.no_grad()
def _to_fp8(x,fp8_dtype):

    fp8_max=torch.finfo(fp8_dtype).max
    amax=x.float().abs().max()

    scale= fp8_max/amax.double().clamp(min=EPS)

    scale=scale.float()

    x_scaled=x.float() * scale
    x_clamped=x_scaled.clamp(-fp8_max,fp8_max)
    x_fp8= x_clamped.to(fp8_dtype)

    inv_scale = scale.reciprocal()

    return x_fp8, inv_scale


def _to_col_major(x):
    return x.t().contiguous().t()

@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):

    @staticmethod
    def forward (ctx, input_2d, weight):

        input_fp8, input_inv= _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        output= torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b = weight_inv,
            out_dtype= input_2d.dtype,
            use_fast_accum=True
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8,w_inv = ctx.saved_tensors

        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)

        w_col=_to_col_major(w_fp8)

        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )


        go_T=go_fp8.t().contiguous()
        in_col = _to_col_major(in_fp8)

        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum= False,

        )



        return grad_input, grad_weight


class Float8Linear(nn.Linear):


    def forward(self,input):

        input= input.to(COMPUTE_DTYPE)

        orig_shape = input.shape

        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d,self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output 


    @classmethod
    def from_float(cls,mod):
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)

        new_mod.weight= mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:

    @staticmethod
    def from_recipe_name(recipe_name):
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'."
                f"Rowwise/axiswise recipes require the full torchao library."
            )

        return Float8LinearConfig()


def convert_to_float8_training(module, *, config=None, module_filter_fn= None):
    
    def _convert(mod,prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child,nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child,fqn):
                    setattr(mod,name,Float8Linear.from_float(child))

        

    _convert(module)
    return module




    



