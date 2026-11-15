use std::net::IpAddr;

/// Checks if the given IP address belongs to the local machine.
pub fn is_local_addr(ip: IpAddr) -> bool {
    if ip.is_loopback() {
        return true;
    }
    match get_if_addrs::get_if_addrs() {
        Ok(interfaces) => interfaces.iter().any(|iface| iface.ip() == ip),
        Err(_) => false, // If we can't get local addresses, assume it's not local.
    }
}
