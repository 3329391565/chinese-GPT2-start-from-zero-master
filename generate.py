import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import transformers
import argparse
from datetime import datetime
from tokenizations.bpe_tokenizer import get_encoder
from tqdm import tqdm

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    logits = logits.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(model, context, length, temperature=1.0, top_k=0, top_p=0.0, progress_desc=None):
    generated = context
    device = context.device
    for _ in tqdm(range(length), desc=progress_desc, ncols=100, ascii=True):
        with torch.no_grad():
            inputs = {"input_ids": generated}
            outputs = model(**inputs)
            if isinstance(outputs, tuple):
                next_token_logits = outputs[0][0, -1, :] / temperature
            else:
                next_token_logits = outputs.logits[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probabilities = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

def generate_samples(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # tokenizer
    if args.bpe_token:
        encoder = get_encoder(args.tokenizer_path, args.vocab_bpe)
        encode = encoder.encode
        decode = encoder.decode
    else:
        from tokenizations import tokenization_bert
        encoder = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
        encoder.max_len = 999999
        encode = lambda x: encoder.convert_tokens_to_ids(encoder.tokenize(x))
        # 修改 decode 保留中文标点（如双引号）
        decode = lambda x: ''.join([encoder._convert_id_to_token(i).replace('##', '') for i in x])

    # model
    model = transformers.GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    # 输出文件
    if args.save_samples:
        if os.path.isdir(args.save_samples_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(args.save_samples_path, f"samples_{timestamp}.txt")
        else:
            output_file = args.save_samples_path
    else:
        output_file = args.output_file

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 清理 prefix
    clean_prefix = args.prefix
    for tok in ['[CLS]', '[MASK]', '[SEP]', '[PAD]']:
        clean_prefix = clean_prefix.replace(tok, '')

    # 生成样本
    with open(output_file, 'w', encoding='utf8') as f:
        for i in range(1, args.num_samples + 1):
            context = torch.tensor([encode(args.prefix)], dtype=torch.long).to(device)
            generated_ids = sample_sequence(
                model=model,
                context=context,
                length=args.length,
                temperature=args.temperature,
                top_k=args.topk,
                top_p=args.topp,
                progress_desc=f"SAMPLE {i}"
            )
            gen_ids = generated_ids[0, context.shape[1]:].tolist()
            text = decode(gen_ids)
            # 清理生成文本中的特殊 token
            for tok in ['[CLS]', '[MASK]', '[SEP]', '[PAD]']:
                text = text.replace(tok, '')
            text = text.replace('\n', '').strip()
            f.write(f"==================== SAMPLE {i} ====================\n")
            f.write(f"{clean_prefix}{text}\n\n")

    print(f"All {args.num_samples} samples saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str)
    parser.add_argument('--vocab_bpe', default='tokenizations/vocab.bpe', type=str)
    parser.add_argument('--bpe_token', action='store_true')
    parser.add_argument('--model_path', default='model/final_model', type=str)
    parser.add_argument('--prefix', default='砰的一声', type=str)
    parser.add_argument('--length', default=200, type=int)
    parser.add_argument('--topk', default=0, type=int)
    parser.add_argument('--topp', default=0.0, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--num_samples', default=10, type=int)
    parser.add_argument('--output_file', default='./mnt/samples.txt', type=str)
    parser.add_argument('--save_samples', action='store_true')
    parser.add_argument('--save_samples_path', default='./mnt/samples.txt', type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    generate_samples(args)

if __name__ == '__main__':
    main()
