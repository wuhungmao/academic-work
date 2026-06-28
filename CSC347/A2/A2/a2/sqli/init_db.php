<?php
$db = new PDO('sqlite:./app.db');
$db->exec("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, email TEXT, password_hash TEXT);");
$db->exec("DELETE FROM users;");
$db->exec("INSERT INTO users (username,email,password_hash) VALUES  ('alice','alice@example.com','\$2y\$10\$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'),('bob','bob@example.com','\$2y\$10\$BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'), ('admin','admin@localhost','\$2y\$10\$CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC');");
echo "DB init done\n";

