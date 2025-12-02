#include "network_interface.hh"
#include "parser.hh"
#include "arp_message.hh"
#include "ethernet_frame.hh"

using namespace std;

// ethernet_address: Ethernet (what ARP calls "hardware") address of the interface
// ip_address: IP (what ARP calls "protocol") address of the interface
NetworkInterface::NetworkInterface( const EthernetAddress& ethernet_address, const Address& ip_address )
  : ethernet_address_( ethernet_address ), ip_address_( ip_address ), arp_table(), 
    cache_time_table(), arp_time_table(), waiting_queue(), ready_queue()
{
  cerr << "DEBUG: Network interface has Ethernet address " << to_string( ethernet_address_ ) << " and IP address "
       << ip_address.ip() << "\n";
}

// dgram: the IPv4 datagram to be sent
// next_hop: the IP address of the interface to send it to (typically a router or default gateway, but
// may also be another host if directly connected to the same network as the destination)

// Note: the Address type can be converted to a uint32_t (raw 32-bit IP address) by using the
// Address::ipv4_numeric() method.

void NetworkInterface::send_datagram( const InternetDatagram& dgram, const Address& next_hop )
{
  auto mac_address = arp_table.find(next_hop.ipv4_numeric());
  if (mac_address != arp_table.end()) {
    //If you know the MAC address, you can create the proper Ethernet frame and
    // put it in the ready-to-be-sent queue (which will be discussed later on). To do this, you should
    // create an Ethernet frame, set the type as IPv4 packet (type = EthernetHeader::TYPE IPv4),
    // properly set the source and destination MAC addresses, and put the serialized version of the
    // dgram in it
    EthernetHeader header;
    header.src = ethernet_address_;
    header.dst = mac_address->second;
    header.type = EthernetHeader::TYPE_IPv4;

    EthernetFrame frame;
    frame.header = header;

    Serializer serializer;
    dgram.serialize(serializer);
    frame.payload = serializer.output();

    ready_queue.push(frame);

  } else {
    //if we have sent arp request to search 
    //for mac address for next_hop's ip address
    //then we simply add packet to waiting
    auto it = arp_time_table.find(next_hop.ipv4_numeric());

    // Check if the IP address exists in arp_time_table
    if (it != arp_time_table.end()) {
      waiting_queue.push_back({dgram, next_hop});
    } else {
      waiting_queue.push_back({dgram, next_hop});

      ARPMessage arp_request;
      arp_request.opcode = ARPMessage::OPCODE_REQUEST;
      arp_request.sender_ethernet_address = ethernet_address_;
      arp_request.target_ethernet_address = ETHERNET_BROADCAST;
      arp_request.sender_ip_address = ip_address_.ipv4_numeric();
      arp_request.target_ip_address = next_hop.ipv4_numeric();
      //we need to broadcast arp request for this ip
      //address to find its mac address. Put the packet in
      //waiting queue
      EthernetHeader header;
      header.src = ethernet_address_;
      header.dst = ETHERNET_BROADCAST;
      header.type = EthernetHeader::TYPE_ARP;

      EthernetFrame frame;
      frame.header = header;

      Serializer serializer;
      arp_request.serialize(serializer);
      frame.payload = serializer.output();

      ready_queue.push(frame);
      arp_time_table.insert( {arp_request.target_ip_address, 0} );
    }
  }
}

// frame: the incoming Ethernet frame
optional<InternetDatagram> NetworkInterface::recv_frame( const EthernetFrame& frame )
{
  if(frame.header.dst != ethernet_address_ && frame.header.dst != ETHERNET_BROADCAST){
    //This frame is not for this machine, discard it
    return {};
  } else if(frame.header.dst == ethernet_address_ && frame.header.type == EthernetHeader::TYPE_IPv4) {
    //This frame is for this machine and its payload is an IPv4 packet. Parse it and return if no error;
    InternetDatagram payload;
    // Parser parser(frame.payload);
    bool parse_result = parse<InternetDatagram>(payload, frame.payload);
    if(parse_result){
      return payload;
    }
  } else if (frame.header.dst == ethernet_address_ && frame.header.type == EthernetHeader::TYPE_ARP){
    //This frame is for this machine, and it is a arp message. we need to parse the arp message
    ARPMessage arp_message;
    // Parser parser(frame.payload);
    bool parse_result = parse<ARPMessage>(arp_message, frame.payload);
    if(parse_result){
      //If it could be parsed properly, then it should learn the mapping between the packet’s Sender
      //IP address and its MAC address and cache this in the ARP cache table. This information 
      //should be cached for 30 seconds. <-- ???

      if(arp_message.opcode == ARPMessage::OPCODE_REPLY) 
      {
        uint32_t sender_ip_address = arp_message.sender_ip_address;
        EthernetAddress sender_ethernet_address = arp_message.sender_ethernet_address;
        arp_table.insert( {sender_ip_address, sender_ethernet_address} );
        arp_time_table.insert( {sender_ip_address, 0} );

        //Since we just update arp_table, we need to process the datagram stored in waiting queue
        size_t queueSize = waiting_queue.size();
        
        // Loop through the waiting queue
        for (size_t i = 0; i < queueSize; ++i) {
            const auto& pair = waiting_queue[i];
            const InternetDatagram& datagram = pair.first;
            const Address& next_hop = pair.second;

            // Call the function with the retrieved pair
            send_datagram(datagram, next_hop);
        }
      }

      else if (arp_message.opcode == ARPMessage::OPCODE_REQUEST) {
        //Furthermore, if it is an ARP request that asks for our IP address, reply back to it. To do this,
        //you should create an ARP reply packet that is destined to the sender and contains proper
        //information (including our IP and MAC address). Then package this ARP message in 
        //an Ethernet frame and place it in the ready-to-be-sent queue.
        if(arp_message.target_ip_address == ip_address_.ipv4_numeric()) 
        {
          //Other nodes are looking for our mac address, hence send the ARP reply packet to them.
          ARPMessage arp_reply;
          arp_reply.opcode = ARPMessage::OPCODE_REPLY;
          arp_reply.sender_ethernet_address = ethernet_address_;
          arp_reply.target_ethernet_address = arp_message.sender_ethernet_address;
          arp_reply.sender_ip_address = ip_address_.ipv4_numeric();
          arp_reply.target_ip_address = arp_message.sender_ip_address;
          //Create ARP reply to send it to the node requesting our Ethernet address.
 
          EthernetHeader header;
          header.src = ethernet_address_;
          header.dst = arp_message.sender_ethernet_address;
          header.type = EthernetHeader::TYPE_ARP;

          EthernetFrame arp_reply_frame;
          arp_reply_frame.header = header;

          Serializer serializer;
          arp_reply.serialize(serializer);
          arp_reply_frame.payload = serializer.output();

          //Added ARP reply to ready_queue
          ready_queue.push(arp_reply_frame);
        }
      }
    }
  }
  return {};
}

// ms_since_last_tick: the number of milliseconds since the last call to this method
//void NetworkInterface::tick( const size t ms since last tick )
// This is the callback function that informs you about the passage of time. When this function
// is called, it means that ms since last tick milliseconds are passed from the last time that it was
// called. You should keep track of time and perform the following two tasks:
// – Expire any entry in ARP cache table that was learnt more than 30 seconds ago.
// – Remove the pending ARP reply wait for any next hop IP that was sent more than 5 seconds
// ago. Furthermore, you should also empty any packets waiting for that IP address from the
// queue.
void NetworkInterface::tick( const size_t ms_since_last_tick )
{
  for (auto& entry : cache_time_table) {
    entry.second += ms_since_last_tick;
    // Check if the entry has expired and remove it if necessary
    if (entry.second >= 30000) { // 30 seconds in milliseconds
        // Entry has expired, remove it from the table
        cache_time_table.erase(entry.first);
    }
  }

  // Increment time for each record in arp_time_table by ms_since_last_tick
  for (auto& entry : arp_time_table) {
    entry.second += ms_since_last_tick;
    // Check if the entry has expired and remove it if necessary
    if (entry.second >= 5000) { // 5 seconds in milliseconds
      // Entry has expired, remove it from the table
      uint32_t ip_address = entry.first;
      arp_time_table.erase(entry.first);
      // Remove any packets waiting for that IP address from the queue
      for (auto it = waiting_queue.begin(); it != waiting_queue.end(); ) {
        if (it->second.ipv4_numeric() == ip_address) {
            // Remove the current entry from the waiting_queue
            it = waiting_queue.erase(it);
        } else {
            // Move to the next entry
            ++it;
        }
      }
    }
  }
}

optional<EthernetFrame> NetworkInterface::maybe_send()
{
  //Whenever the physical layer of the network is ready to send out a packet, it will call this function
  //to check if there is any packet ready to be sent. This is where you should check your 
  //ready-to-be-sent queue to see if there is any packet in it. If there any packet, you
  //should remove the first packet waiting in the queue (the oldest packet) from it and return it.
  //Otherwise, if there is no packet to be sent, simply return nothing.
  //Note that there could be different Ethernet packets in the queue: datagrams that are passed by
  //IP layer, ARP requests to learn the MAC address of a next hop, and ARP replies to requests that
  //are sent to us about our IP addresses.
  if (!ready_queue.empty()) {
    EthernetFrame frame = ready_queue.front();
    ready_queue.pop();
    return frame;
  } else {
    return std::nullopt;
  }
}
