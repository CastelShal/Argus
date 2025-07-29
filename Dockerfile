FROM python:3.10.18-bookworm AS builder
WORKDIR /app

#install cmake
RUN apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

# replace opencv with the headless variant / avoids x11 conflicts
RUN sed -i 's/opencv-python/opencv-python-headless/' requirements.txt
RUN sed -i 's/opencv-contrib-python/opencv-contrib-python-headless/' requirements.txt

#pip install
RUN pip install --no-deps --no-cache -r requirements.txt

# Build stage
FROM python:3.10.18-slim

# Install only necessary runtime libraries (based on ldd)
RUN apt-get update && apt-get install -y --no-install-recommends \
libx11-6 \
libpng16-16 \
libjpeg62-turbo \
libwebp7 \
libstdc++6 \
libgcc1 \
libc6 \
libxcb1 \
zlib1g \
libxau6 \
libxdmcp6 \
libbsd0 \
libmd0 \
&& rm -rf /var/lib/apt/lists/*

# Copy all libraries
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy app folder
COPY --from=builder /app /app

WORKDIR /app

EXPOSE 5000

CMD python src/train.py ; exec python src/main.py