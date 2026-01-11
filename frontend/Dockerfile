# Node base image
FROM node:18-alpine

WORKDIR /app

# Install React dependencies
COPY frontend/package.json .
RUN npm install react-scripts

RUN npm install

# Copy source
COPY frontend/public ./public
COPY frontend/src ./src

EXPOSE 3000

CMD ["npm", "start"]
