import os

# 修复 OpenMP 重复加载问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 推荐避免显存碎片化（可在某些 PyTorch 版本下缓解 OOM）
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 修复 requests 字符编码依赖警告
try:
    import chardet
except ImportError:
    print("chardet 未安装，可通过 `pip install chardet` 安装")
try:
    import charset_normalizer
except ImportError:
    print("charset_normalizer 未安装，可通过 `pip install charset_normalizer` 安装")

import transformers
import torch
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from tokenizations.bpe_tokenizer import get_encoder

# Import model/config from transformers
from transformers import GPT2LMHeadModel, GPT2Config

# Try to import scheduler and AdamW from different locations for compatibility
try:
    # newer transformers may expose optimization helpers under optimization
    from transformers.optimization import get_linear_schedule_with_warmup, AdamW as HfAdamW
except Exception:
    try:
        # older versions exposed them at top-level
        from transformers import get_linear_schedule_with_warmup, AdamW as HfAdamW
    except Exception:
        get_linear_schedule_with_warmup = None
        HfAdamW = None

# Also prefer torch's AdamW when available
try:
    from torch.optim import AdamW as TorchAdamW
except Exception:
    TorchAdamW = None



def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
    all_len = len(lines)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    print('finish')


def main():
    parser = argparse.ArgumentParser()
    # 默认不强制指定 CUDA 设备；如果想使用 GPU，请传入 e.g. --device 0 或 --device 0,1
    parser.add_argument('--device', default='', type=str, required=False, help='设置使用哪些显卡，例如 "0" 或 "0,1"，不传则默认使用 CPU')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    # Only set CUDA_VISIBLE_DEVICES if the user explicitly requested devices.
    # This avoids accidentally masking CUDA when the default should be CPU.
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    # Parse requested device ids (e.g. "0" or "0,1") and set primary device.
    requested_devices = [d.strip() for d in args.device.split(',') if d.strip() != '']
    cuda_available = torch.cuda.is_available() and len(requested_devices) > 0
    if cuda_available:
        # choose the first requested device as the primary CUDA device
        try:
            primary_cuda_id = int(requested_devices[0])
        except Exception:
            primary_cuda_id = 0
        # set the torch default device
        torch.cuda.set_device(primary_cuda_id)
        device = torch.device(f'cuda:{primary_cuda_id}')
        print('using device:', device, 'requested_devices=', requested_devices, 'cuda_count=', torch.cuda.device_count())
    else:
        device = torch.device('cpu')
        if args.device and not torch.cuda.is_available():
            print('requested CUDA device(s) but CUDA is not available in this environment.\n' \
                  'Please install a CUDA-enabled build of PyTorch or run without --device to use CPU.')
        else:
            print('using device: cpu (no CUDA available or no device requested)')

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw  # 选择是否从零开始构建数据集
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    # Print resolved output directory to avoid accidental overwrites
    resolved_output_dir = os.path.abspath(output_dir)
    print(f'Using output_dir: {resolved_output_dir}')
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if raw:
        print('building files')
        build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
                    full_tokenizer=full_tokenizer, min_length=min_length)
        print('files built')

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)

    # Try to reduce memory footprint: disable generation cache and enable gradient checkpointing if supported
    try:
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
            print('Disabled model.config.use_cache to reduce memory')
    except Exception:
        pass
    try:
        # gradient_checkpointing_enable exists in newer transformers and reduces memory at cost of compute
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print('Enabled gradient checkpointing to save memory')
    except Exception:
        pass

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    print('calculating total steps (exact per-piece counts)')
    # Compute exact number of batches per piece using the same sliding-window logic as training.
    steps_per_epoch = 0
    for i in tqdm(range(num_pieces)):
        piece_path = os.path.join(tokenized_data_path, f'tokenized_train_{i}.txt')
        try:
            with open(piece_path, 'r') as f:
                tokens = f.read().strip().split()
        except FileNotFoundError:
            print(f'Warning: tokenized file not found: {piece_path}, skipping')
            continue
        L = len(tokens)
        # reproduce sample extraction logic used in training loop
        samples = 0
        start_point = 0
        if L <= n_ctx:
            samples = 1
        else:
            while start_point < L - n_ctx:
                samples += 1
                start_point += stride
            if start_point < L:
                samples += 1
        # number of batch steps (we drop the last incomplete batch in training loop)
        batches_in_piece = samples // batch_size
        steps_per_epoch += batches_in_piece

    # total batch-level steps across all epochs, and optimizer steps (considering gradient accumulation)
    total_batch_steps = int(steps_per_epoch * epochs)
    total_optimizer_steps = int(total_batch_steps // gradient_accumulation)
    print(f'total batch steps = {total_batch_steps} (batch-level steps across all epochs)')
    print(f'total optimizer steps = {total_optimizer_steps} (used for LR scheduler)')
    print(f'steps per epoch (exact batches) = {steps_per_epoch}')

    # Choose AdamW implementation: prefer torch's, then transformers', otherwise use torch.optim.Adam
    if TorchAdamW is not None:
        optimizer = TorchAdamW(model.parameters(), lr=lr)
    elif HfAdamW is not None:
        optimizer = HfAdamW(model.parameters(), lr=lr, correct_bias=True)
    else:
        # last resort: use SGD (not ideal) or the basic torch optimizer; here we use Adam as fallback
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=lr)

    # Create scheduler: prefer transformers' get_linear_schedule_with_warmup, otherwise fallback to LambdaLR
    if get_linear_schedule_with_warmup is not None:
        # Use optimizer-step level count for scheduler (total_optimizer_steps)
        try:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_optimizer_steps)
        except Exception:
            # fallback if total_optimizer_steps not defined yet
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=1)
    else:
        # simple linear warmup then constant lr via LambdaLR
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)
    use_native_amp = False
    scaler = None
    if fp16:
        # prefer native AMP if available
        if hasattr(torch.cuda, 'amp'):
            # Newer PyTorch prefers torch.amp.autocast (device-aware) while older
            # versions used torch.cuda.amp.autocast. Try to use torch.amp.autocast
            # and fall back to torch.cuda.amp.autocast for compatibility.
            try:
                from torch.amp import autocast
            except Exception:
                from torch.cuda.amp import autocast
            from torch.cuda.amp import GradScaler
            use_native_amp = True
            scaler = GradScaler()
            print('Using native torch.cuda.amp/torch.amp for FP16')
        else:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex or use a PyTorch with native amp to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    # If multiple CUDA devices were requested and available, wrap with DataParallel
    multi_gpu = False
    available_cuda_count = torch.cuda.device_count()
    parsed_device_ids = []
    if device.type == 'cuda' and len(requested_devices) > 1 and available_cuda_count > 1:
        # map requested ids to integers and keep only those < available_cuda_count
        for d in requested_devices:
            try:
                di = int(d)
            except Exception:
                continue
            if di < available_cuda_count:
                parsed_device_ids.append(di)
        if len(parsed_device_ids) > 1:
            print("Let's use", len(parsed_device_ids), "GPUs! device_ids=", parsed_device_ids)
            model = DataParallel(model, device_ids=parsed_device_ids)
            multi_gpu = True
    print('starting training')
    # batch_global_step counts batches processed (used for batch-based logging)
    batch_global_step = 0
    # opt_step counts optimizer.step() calls (used for scheduler and LR)
    opt_step = 0
    running_loss = 0
    epoch_start_overall = 0
    epoch_step_counter = 0
    for epoch in range(epochs):
        # clearer epoch header for easier log parsing
        print('\n' + '=' * 10 + f' EPOCH {epoch + 1} START ' + '=' * 10)
        now = datetime.now()
        print('time: {}'.format(now))
        # record the batch_global_step at the start of this epoch to compute accurate step-in-epoch
        epoch_start_overall = batch_global_step
        # reset per-epoch step counter
        epoch_step_counter = 0
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        for i in x:
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                line = f.read().strip()
            tokens = line.split()
            tokens = [int(token) for token in tokens]
            start_point = 0
            samples = []
            while start_point < len(tokens) - n_ctx:
                samples.append(tokens[start_point: start_point + n_ctx])
                start_point += stride
            if start_point < len(tokens):
                samples.append(tokens[len(tokens)-n_ctx:])
            random.shuffle(samples)
            for step in range(len(samples) // batch_size):  # drop last

                #  prepare data
                batch = samples[step * batch_size: (step + 1) * batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)

                #  forward pass
                if use_native_amp:
                    # prefer device-aware autocast context (torch.amp.autocast or
                    # torch.cuda.amp.autocast) to avoid FutureWarning
                    try:
                        # torch.amp.autocast accepts device_type arg
                        with autocast(device_type='cuda'):
                            outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                            loss, logits = outputs[:2]
                    except TypeError:
                        # older autocast signature may not accept device_type
                        with autocast():
                            outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                            loss, logits = outputs[:2]
                else:
                    outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                    loss, logits = outputs[:2]

                #  get loss
                if multi_gpu:
                    loss = loss.mean()
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

                #  loss backward
                if use_native_amp:
                    scaler.scale(loss).backward()
                    # unscale will be called once before optimizer.step() to avoid multiple unscale calls
                    # when using gradient accumulation
                elif fp16 and 'amp' in globals():
                    # apex
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                #  optimizer step
                if (batch_global_step + 1) % gradient_accumulation == 0:
                    running_loss += loss.item()
                    # perform optimizer step (handle amp and non-amp paths)
                    if use_native_amp:
                        try:
                            scaler.unscale_(optimizer)
                        except RuntimeError:
                            pass
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    # zero grad, step scheduler and record optimizer step count
                    optimizer.zero_grad()
                    try:
                        scheduler.step()
                    except Exception:
                        pass
                    opt_step += 1

                    # free up cached memory to reduce fragmentation/peak usage
                    try:
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                if (batch_global_step + 1) % log_step == 0:
                    tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, batch_global_step)
                    # More informative logging: include overall step, estimated epoch and step-in-epoch, and seconds
                    display_batch = batch_global_step + 1
                    display_opt = opt_step + 1
                    try:
                        epoch_now = epoch + 1
                        # epoch_step_counter reflects number of completed steps in this epoch so far,
                        # printing uses +1 to indicate the current step (since overall_step hasn't been
                        # incremented yet at this point in code).
                        step_in_epoch = epoch_step_counter + 1
                    except Exception:
                        step_in_epoch = step + 1
                        epoch_now = epoch + 1
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(f'now time: {ts}. BatchStep {display_batch}/{total_batch_steps} | OptStep {display_opt}/{total_optimizer_steps}. Step {step+1} of piece {piece_num} (epoch {epoch_now} step-in-epoch {step_in_epoch}), loss {running_loss * gradient_accumulation / (log_step / gradient_accumulation)}')
                    running_loss = 0
                # increment batch and per-epoch counters after processing this batch
                batch_global_step += 1
                epoch_step_counter += 1
            piece_num += 1

        print('saving model for epoch {}'.format(epoch + 1))
        # build a safe, joined directory path and ensure it exists
        save_dir = os.path.join(output_dir, f'model_epoch{epoch + 1}')
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(save_dir)
        print(f'model saved to: {save_dir}')
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    final_save_dir = os.path.join(output_dir, 'final_model')
    os.makedirs(final_save_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(final_save_dir)
    print(f'final model saved to: {final_save_dir}')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()
