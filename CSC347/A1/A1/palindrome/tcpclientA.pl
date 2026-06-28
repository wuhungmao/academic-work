#!/usr/bin/perl
#tcpclient.pl

use IO::Socket;

$socket = new IO::Socket::INET (
                                  PeerAddr  => '10.128.10.25',
                                  PeerPort  =>  7778,
                                  Proto => 'tcp',
                               )                
or die "Couldn't connect to Server\n";
# (gdb) print secret
# $1 = 0x8048720 "you got me!!"
$count = 0;
while ($count < 1100) {
    $socket->recv($recv_data,1024);
	    print "RECIEVED: $recv_data"; 
        
        $str_addr = "Z" x $count;
        $send_data =  $str_addr . "\n";
        
        $tmp=$send_data;
	chop($tmp); # get rid of new line
        
        if ($tmp ne 'quit') {
	        $socket->send($send_data);
        }    else {
	        $socket->send($send_data);
            	close $socket;
            	last;
        }
        # print "Count is: $count\n";
        $count++;
        print "count is " . $count . "\n";
}
