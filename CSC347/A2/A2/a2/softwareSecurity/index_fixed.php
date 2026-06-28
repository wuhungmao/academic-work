<?
        session_start();

        ###################################################################################################
        # https://www.owasp.org/index.php/Top_10_2013-Top_10
        ###################################################################################################

        ###################################################################################################
        # FIX OWASP A8: CSRF (Cross Site Request Forgery)
        # First verify that this is vulnerable. That is, if attacker gets target to follow a malicious
        # URL, the attacker can get the user to perform an operation they had not intended.
        # For example, craft a URL that gets the target to delete one of their entries.
        # Fix this by creating a random page token. The page token is placed in the
        # session. Now every reply to a page, includes the page token, in the URL, as a hidden variable etc.
        # Before processing a request, first verify the page token.
        ###################################################################################################
        if(!isset($_SESSION['csrfToken'])){
                $_SESSION['csrfToken'] = bin2hex(random_bytes(16));
        }

        if($_SERVER['REQUEST_METHOD'] === "POST" || (isset($_REQUEST['operation']) && $_REQUEST['operation'] === "deleteExpression")) {
                $token = $_SERVER['REQUEST_METHOD'] === "POST" ? $_POST['csrfToken'] : $_REQUEST['csrfToken'];
                
                // do validation
                if(!isset($token) || !hash_equals($_SESSION['csrfToken'], $token)){
                        die("Potential CSRF attack detected");
                }
        }
        
        if(!isset($_SESSION['isLoggedIn']))$_SESSION['isLoggedIn']=False;
        $operation=$_REQUEST['operation'];
        $g_debug="";
        $g_errors="";

        // overridden database functions
        function pg_connect_db(){

                ###################################################################################################
                # FIX OWASP A6: SENSITIVE DATA EXPOSURE
                # Take a look at the db, whats wrong with the passwords?
                # Fix by hashing passwords, as well, you need to fix authentication to check hashes
                # Additionally, the application is not using https to communicate, fix this
                ###################################################################################################
                // prof told me we can assume the passwords stored in the database is hashed already
                $password = "adg135sfh246";
                $dbconn = pg_connect("dbname=fourfours user=ff host=localhost password=$password");
                pg_set_client_encoding($dbconn, 'UTF8');
                return $dbconn;
        }
        // need to escape output to prevent XSS
        function escape($output) {
                return htmlspecialchars($output, ENT_QUOTES, 'UTF-8');
        }
        if($operation == "login"){
                $user = $_REQUEST['user'];
                $password = $_REQUEST['password'];
                $dbconn = pg_connect_db();

                // FIX OWASP A1: SQL Injection is handled by prepared statements.
                $query = "SELECT id, username, firstName, lastName, passwd FROM account WHERE username=$1";
                
                $result = pg_prepare($dbconn, "login_data", $query); 
                $result = pg_execute($dbconn, "login_data", array($user));
                
                // Fetch the row to access all fields
                if($row = pg_fetch_row($result)) {
                        $stored_hashed_password = $row[4]; 
                        
                        if (password_verify($password, $stored_hashed_password)) {
                        
                                session_regenerate_id(true); // FIX OWASP A2: SESSION FIXATION

                                $_SESSION['accountId'] = $row[0];
                                $_SESSION['user'] = $row[1];
                                $_SESSION['firstName'] = $row[2];
                                $_SESSION['lastName'] = $row[3];
                                $_SESSION['isLoggedIn'] = True;
                        } else {
                                $g_debug = "$user Invalid credentials."; 
                                $_SESSION['isLoggedIn'] = False;
                        }
                } else {
                        $g_debug = "$user Invalid credentials."; 
                        $_SESSION['isLoggedIn'] = False;
                }
        } elseif ($operation == "deleteExpression"){
                $expressionId = $_REQUEST['expressionId'];
                $accountId=$_SESSION['accountId'];
                $dbconn = pg_connect_db();

                ###################################################################################################
                # FIX OWASP 2013 A4: INSECURE DIRECT OBJECT REFERENCES
                # Prove that it is vulnerable by logging in as one user and deleting another users entry
                # Fix this by...
                # Either fix the insecure part, that is, verify that the user can perform the operation
                # of the direct object reference part, that is, fix the id's so they don't directly
                # reference the expressionId, or both (even better).
                # Another problem: why get account id from the request? In this case, this is part of the
                # insecure direct object reference, that is, referencing the account id.
                # Note: Simply not giving the user interface the option to delete is not sufficient.
                ###################################################################################################
                   // Step 1: Verify that the expression belongs to the logged-in user
                $verifyQuery = "SELECT 1 FROM solution WHERE id = $1 AND accountId = $2";
                $verifyStatement = pg_prepare($dbconn, "verify", $verifyQuery);
                $verifyResult = pg_execute($dbconn, "verify", array($expressionId, $accountId));

                if(pg_num_rows($verifyResult) === 0) {
                        die("Unauthorized attempt to delete expression.");
                }
                $deleteQuery = "DELETE FROM solution WHERE id=$1 AND accountId=$2";
                $deleteStatement = pg_prepare($dbconn, "delete", $deleteQuery);
                $deleteResult = pg_execute($dbconn, "delete", array($expressionId, $accountId));

        } elseif ($operation == "addExpression"){

                ###################################################################################################
                # FIX: XSS: user input/output is not vetted
                # First check that the application is vulnerable by placing html in the
                # database and then viewing the HTML as it exits the db
                # Fix this by ...
                # Either whitelisting the input, or escape the input
                # Do the same for all untrusted input and output!
                # http://stackoverflow.com/questions/46483/htmlentities-vs-htmlspecialchars
                ###################################################################################################

                $expression = htmlspecialchars($_REQUEST['expression'], ENT_QUOTES, 'UTF-8');
                $value = htmlspecialchars($_REQUEST['value'], ENT_QUOTES, 'UTF-8');
                $accountId=$_SESSION['accountId'];

                $dbconn = pg_connect_db();
                $result = pg_prepare($dbconn, "check_add", "SELECT * FROM solution WHERE expression= $1");
                $result = pg_execute($dbconn, "check_add", array($expression));
                if(!($row = pg_fetch_row($result))) {
                        $insertQuery = "INSERT INTO solution (value, expression, accountId) VALUES ($1, $2, $3)";
                        $result = pg_prepare($dbconn, "add", $insertQuery);
                        $result = pg_execute($dbconn, "add", array($value, $expression, $accountId));
                } else {
                        $g_errors="$expression is already in our database";
                }
        } elseif($operation == "logout") {
                unset($_SESSION);
                $_SESSION['isLoggedIn']=False;
        }
        $g_isLoggedIn=$_SESSION['isLoggedIn'];
        $g_index="";
        for($i=0;$i<=100;$i+=10){ $g_index=$g_index . "<a href=#$i>$i</a> "; }
        $g_userFullName=$_SESSION['firstName'] . " " . $_SESSION['lastName'];
        $g_userFirstName=$_SESSION['firstName'];
        $g_accountId=$_SESSION['accountId'];
?>

<html>
        <body>
                <center>
                <h1>Four Fours</h1>
                <font color="red"><?=$g_errors ?></font><br/><br/>
                <? if($g_isLoggedIn){ ?>
                        <a href=?operation=logout>Logout</a>
                        <br/>
                        <br/>
                        <div style="width:400px; text-align:left;">
                        Welcome <?= $g_userFirstName ?>.
                        Using only four 4s' and the operations +,-,*,/,^ (=exponentiation) and sqrt (=square root)
                        create as many of the values below as you can. For example, for 2, I have ((4/4)+(4/4)), for 16, I have sqrt(4*4*4*4).
                        </div>
                        <br/>
                        <table>
                                <tr>
                                        <th>value</th><th>expression and author</th>
                                </tr>
                                <?php
                                for($i=0;$i<=100;$i++){
                                        if($i%10==0){ ?>
                                                <td align="center" colspan="2" style="border-bottom:2pt solid black;"><?=$g_index ?></td>
                                        <? } ?>

                                        <tr>
                                                <td valign="top" style="border-bottom:2pt solid black;"> <a name="<?=$i?>" ><?=$i ?></a></td>
                                                <td valign="top" style="border-bottom:2pt solid black;">
                                                        <table>
                                                                <?php
                                                                        $dbconn = pg_connect_db();
                                                                        $result = pg_prepare($dbconn, "info", "SELECT firstName, lastName, value, expression, s.accountId, s.id FROM account a, solution s WHERE a.id=s.accountId AND value=$i ORDER BY firstName, lastName, expression");
                                                                        $result = pg_execute($dbconn, "info", array());
                                                                        # FIX XSS: Output from users must be whitelisted or escaped

                                                                        while ($row = pg_fetch_row($result)) {
                                                                                $count=0;
                                                                                $firstName=$row[$count++];
                                                                                $lastName=$row[$count++];
                                                                                $value=$row[$count++];
                                                                                $expression=$row[$count++];
                                                                                $expressionAccountId=$row[$count++];
                                                                                $expressionId=$row[$count++];
                                                                                if($expressionAccountId==$g_accountId){
                                                                                        // put csrf token in the delete link?
                                                                                        // Retrieve the current token from the session
                                                                                        $csrfToken = $_SESSION['csrfToken'];

                                                                                        // Embed the token in the URL as a GET parameter
                                                                                        $deleteLink="<a href=\"?operation=deleteExpression&expressionId=$expressionId&csrfToken=$csrfToken&accountId=$g_accountId\"><img src=\"delete.png\" width=\"20\" border=\"0\" /></a>";
                                                                                } else {
                                                                                        $deleteLink="";
                                                                                }
                                                                                echo "<tr><td>" . escape($expression) . "</td><td>$deleteLink</td><td>" . escape($firstName) . " " . escape($lastName) . "</td></tr>";
                                                                        }
                                                                ?>
                                                                <tr>
                                                                        <form method="post">
                                                                                <td><input type="text" name="expression"/> </td>
                                                                                <td><input type="submit" value="add"/></td>
                                                                                <input type="hidden" name="value" value="<?=$i?>"/>
                                                                                <input type="hidden" name="operation" value="addExpression"/>
                                                                                <input type="hidden" name="accountId" value="<?=$g_accountId ?>"/>
                                                                                <input type="hidden" name="csrfToken" value="<?=$_SESSION["csrfToken"] ?>"/>
                                                                        </form>
                                                                </tr>
                                                        </table>
                                                </td>
                                        </tr>
                                <? } ?>
                        </table>
                <? } else { ?>
                        <form method="post">
                                <table>
                                        <tr>
                                                <td>user name: <input type="text" size="10" name="user"/></td>
                                                <td>password: <input type="password" size="10" name="password"/> </td>
                                                <td>
                                                        <input type="hidden" name="operation" value="login"/>
                                                        <input type="submit" value="login"/>
                                                        <input type="hidden" name="csrfToken" value="<?=$_SESSION["csrfToken"] ?>"/>
                                                </td>
                                        </tr>
                                        <tr>
                                                <td colspan="3"><?php echo($g_debug); ?></td>
                                        </tr>
                                </table>
                        </form>
                <? } ?>
                </center>
        </body>
</html>