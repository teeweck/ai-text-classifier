# Node base image
FROM node:18-alpine

ENV MODEL_VERSION=v1

WORKDIR /app

# Install React dependencies
COPY package.json .
RUN npm install react-scripts

RUN npm install

# Copy source
COPY public ./public
COPY src ./src

EXPOSE 3000

CMD ["npm", "start"]
