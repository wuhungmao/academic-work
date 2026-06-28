<?php
// search.php - vulnerable example
$db = new PDO('sqlite:./app.db');

// simple search: supports ?q=TERM
$q = isset($_GET['q']) ? $_GET['q'] : '';
$q_trim = trim($q);

if (!isset($_GET['q']) || $q_trim === '' || $q === '') {
    $query = "SELECT id, username, email FROM users WHERE username = 'andi';";
} else {
    $query = "SELECT id, username, email FROM users WHERE username LIKE '%$q%' OR email LIKE '%$q%';";
}



try {
    $rows = $db->query($query);
} catch (Exception $e) {
    die("Query failed: " . htmlspecialchars($e->getMessage()));
}

?>
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Search</title></head>
<body>
  <h1>Search users</h1>
  <form method="get">
    <input name="q" value="<?php echo htmlspecialchars($q); ?>" />
    <button type="submit">Search</button>
  </form>

  <h2>Results</h2>
  <ul>
  <?php
  if ($rows) {
      foreach ($rows as $r) {
          echo "<li>" . htmlspecialchars($r['username']) . " — " . htmlspecialchars($r['email']) . "</li>";
      }
  } else {
      echo "<li>No results</li>";
  }
  ?>
  </ul>
</body>
</html>

