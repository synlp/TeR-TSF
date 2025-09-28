export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --repo-type dataset --resume-download afeng/MTBench_finance_news --local-dir /data2/user2/MTBench/finance/text

huggingface-cli download --repo-type dataset --resume-download afeng/MTBench_finance_stock --local-dir /data2/user2/MTBench/finance/ts

huggingface-cli download --repo-type dataset --resume-download afeng/MTBench_weather_news --local-dir /data2/user2/MTBench/weather/text

huggingface-cli download --repo-type dataset --resume-download afeng/MTBench_weather_temperature --local-dir /data2/user2/MTBench/weather/ts