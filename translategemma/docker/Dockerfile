FROM vllm/vllm-openai:v0.11.0

WORKDIR /app

ENTRYPOINT ["vllm"]
CMD ["serve", "/models/translategemma-4b-it", "--host", "0.0.0.0", "--port", "8020", "--served-model-name", "translategemma"]
