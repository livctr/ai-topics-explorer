# Use Node.js LTS as the base image
FROM node:23

# Set working directory
WORKDIR /app

# Copy package.json and install dependencies for both frontend and backend
COPY backend/package.json frontend/package.json ./
RUN npm install

# Copy backend and frontend code
COPY backend /app/backend
COPY frontend /app/frontend

# Build frontend
WORKDIR /app/frontend
RUN npm run build

# Set working directory back to root
WORKDIR /app

# Expose necessary ports
EXPOSE 5173

# Start both backend and frontend
CMD ["sh", "-c", "cd backend && npm start & cd ../frontend && npm start"]
