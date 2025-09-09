# --- build stage ---
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
# Prefer reproducible installs; fall back if lockfile incompatible
RUN npm ci || npm install
COPY . .
RUN npm run build

# --- runtime stage ---
FROM nginx:alpine AS runtime
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
