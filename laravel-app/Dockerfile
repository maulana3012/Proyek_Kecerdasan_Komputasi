FROM php:8.4-fpm
WORKDIR /var/www
RUN apt-get update && apt-get install -y build-essential libpng-dev libjpeg62-turbo-dev libfreetype6-dev zip unzip git curl libonig-dev libzip-dev
RUN docker-php-ext-install pdo_mysql mbstring zip exif pcntl
RUN docker-php-ext-configure gd --with-freetype=/usr/include/ --with-jpeg=/usr/include/
RUN docker-php-ext-install gd
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer
COPY . /var/www
EXPOSE 9000
CMD ["php-fpm"]
