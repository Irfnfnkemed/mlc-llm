Run the test script with:

```bash
mlc_llm serve <MODEL> --enable-debug --mode server

python ./eval/__main__.py --tokenizer ...  --dataset ... --dataset-path ... \
--num-gpus ... --num-requests ... --host ... --port ... --request-rate inf \
--num-warmup-requests ... --api-endpoint openai-chat --bench-output ... \
--generate-output ... --final-output ...  [--use-stag]
```

Draw the output figure with:

```bash
python ./eval/draw.py --data-root ...
```