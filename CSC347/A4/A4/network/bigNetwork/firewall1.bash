#!/bin/bash
IPT="/sbin/iptables"

# Flush all tables
$IPT -F # filter is the default table
$IPT -F -t nat
$IPT -F -t mangle

# State rules to make life easier
$IPT -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
$IPT -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
$IPT -A FORWARD -m state --state ESTABLISHED,RELATED -j ACCEPT

    # (a) The only publicly routable ip address in the organization is 93.184.216.34,
    #     this means that all of the hosts on the 192.168.*.* network share the one
    #     public IP 93.184.216.34

# NO RULE NEEDED HERE                                                                         #(a)

    # (b) As far as the internet is concerned, the organization is running a web server
    #     at IP 93.184.216.34. All http traffic received at Firewall1 is forwarded to
    #     192.168.10.100.

# NO RULE NEEDED HERE                                                                         #(b)

    # (c) As far as the internet is concerned, the organization is running a mail server
    #     at IP 93.184.216.34. All smtp traffic received at Firewall1 is forwarded to
    #     192.168.10.25.
    
# NO RULE NEEDED HERE                                                                         #(c)

    # (d) Firewall1 allows ssh in, but only from the admins office machine
    #     192.168.11.19, and from the admins home machine 128.2.2.17.

# NO RULE NEEDED HERE                                                                         #(d)
    
    # (e) The ceo can, from their home machine (34.14.10.18) rdp into their office
    #     desktop (192.168.11.18) using port 3389 on Firewall1.

# NO RULE NEEDED HERE                                                                         #(e)
    
    # (f) For convenience, the admin can ssh into their office machine via...
    #     ssh -p 2222 93.184.216.34 # only works from 128.2.2.17

# NO RULE NEEDED HERE                                                                         #(f)

    # Similarly, the ceo can ssh into their ceo desktop via...
    # ssh -p 2222 93.184.216.34 # only works from 34.14.10.18

# NO RULE NEEDED HERE                                                                         #(f)

    # (g) The mail server at 192.168.10.25 is accessible from the LAN.

# NO RULE NEEDED HERE                                                                         #(g)
    
    # (h) The web server (192.168.10.100) is running some web applications, it is accessible
    #     from the LAN. 

# NO RULE NEEDED HERE                                                                         #(h)

    # (i) To get its work done, the web server needs to connect to the postgresql db server at 192.168.11.100
    #     No other connection into the db server are allowed.
    
# NO RULE NEEDED HERE                                                                         #(i)

    # (j) All LAN machines can access the internet, but are restricted to web (http and https) traffic only. 
    #     So, for example, none of the LAN machines can ssh out past firewall1.

# NO RULE NEEDED HERE                                                                         #(j)

    # (k) No other access to any hosts is allowed.

# NO RULE NEEDED HERE                                                                         #(k)

    # (l) 17.17.17.17 has been found to be attacking the organizations systems. Access
    #     to all services from this IP is denied. 

# NO RULE NEEDED HERE                                                                         #(l)

    # (m) As a precaution, in case 17.17.17.17 has compromised one of our systems, 
    #     no outgoing connections to 17.17.17.17 are allowed.

# NO RULE NEEDED HERE                                                                         #(m)


