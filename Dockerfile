FROM denoland/deno:2.1.4
WORKDIR /app
COPY . .
RUN deno cache worker.ts
CMD ["run", "-A", "worker.ts"]