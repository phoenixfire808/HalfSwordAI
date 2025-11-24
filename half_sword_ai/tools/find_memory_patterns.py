"""
Memory Pattern Finder - Helper script to find AOB patterns for Half Sword

This script helps you find memory patterns using Cheat Engine-style scanning.
Once patterns are found, add them to config.MEMORY_PATTERNS.

Usage:
1. Run Half Sword game
2. Use Cheat Engine to find health/stamina/position addresses
3. Right-click address -> "Find out what accesses this address"
4. Get the AOB pattern from the disassembly
5. Add pattern to config.MEMORY_PATTERNS

Example pattern format: "48 8B 05 ?? ?? ?? ?? 48 85 C0 74 ?? 48 8B 48 08"
Use ?? for wildcard bytes (addresses that change)
"""

import pymem
import pymem.process
from half_sword_ai.config import config

def find_process():
    """Find Half Sword process"""
    try:
        process_names = [
            config.GAME_PROCESS_NAME,
            "HalfSwordUE5-Win64-Shipping.exe",
            "HalfSwordUE5.exe",
            "HalfSword-Win64-Shipping.exe",
        ]
        
        for name in process_names:
            try:
                pm = pymem.Pymem(name)
                print(f"✅ Found process: {name}")
                print(f"   Base address: {hex(pm.base_address)}")
                return pm
            except pymem.exception.ProcessNotFound:
                continue
        
        print("❌ Process not found. Make sure Half Sword is running.")
        return None
    except Exception as e:
        print(f"❌ Error finding process: {e}")
        return None

def scan_for_value(process, value, value_type='float'):
    """
    Scan memory for a specific value
    
    Args:
        process: Pymem process object
        value: Value to search for
        value_type: 'float', 'int', 'int64', 'int32'
    
    Returns:
        List of addresses where value was found
    """
    if not process:
        return []
    
    addresses = []
    try:
        modules = list(process.list_modules())
        
        for module in modules[:5]:  # Limit to first 5 modules for speed
            try:
                module_info = pymem.process.module_from_name(process.process_handle, module.name)
                if not module_info:
                    continue
                
                base = module_info.lpBaseOfDll
                size = min(module_info.SizeOfImage, 0x1000000)  # Limit to 16MB
                
                print(f"Scanning {module.name} ({hex(base)} - {hex(base + size)})...")
                
                # Read memory in chunks
                chunk_size = 0x10000  # 64KB chunks
                for offset in range(0, size, chunk_size):
                    try:
                        chunk = pymem.memory.read_bytes(
                            process.process_handle,
                            base + offset,
                            min(chunk_size, size - offset)
                        )
                        
                        # Search for value in chunk
                        if value_type == 'float':
                            import struct
                            value_bytes = struct.pack('f', float(value))
                            for i in range(len(chunk) - 3):
                                if chunk[i:i+4] == value_bytes:
                                    addresses.append(base + offset + i)
                        
                        elif value_type in ['int', 'int32']:
                            import struct
                            value_bytes = struct.pack('i', int(value))
                            for i in range(len(chunk) - 3):
                                if chunk[i:i+4] == value_bytes:
                                    addresses.append(base + offset + i)
                        
                        elif value_type == 'int64':
                            import struct
                            value_bytes = struct.pack('q', int(value))
                            for i in range(len(chunk) - 7):
                                if chunk[i:i+8] == value_bytes:
                                    addresses.append(base + offset + i)
                    
                    except Exception as e:
                        continue
                
            except Exception as e:
                continue
        
        return addresses
    
    except Exception as e:
        print(f"Error scanning: {e}")
        return []

def get_pattern_from_address(process, address, length=20):
    """
    Get AOB pattern from memory address
    
    Args:
        process: Pymem process object
        address: Memory address
        length: Number of bytes to read
    
    Returns:
        AOB pattern string
    """
    if not process:
        return None
    
    try:
        bytes_data = pymem.memory.read_bytes(process.process_handle, address, length)
        pattern_parts = []
        
        for byte_val in bytes_data:
            pattern_parts.append(f"{byte_val:02X}")
        
        return " ".join(pattern_parts)
    
    except Exception as e:
        print(f"Error reading pattern: {e}")
        return None

def main():
    """Main function - interactive pattern finder"""
    print("=" * 60)
    print("Half Sword Memory Pattern Finder")
    print("=" * 60)
    print()
    
    process = find_process()
    if not process:
        return
    
    print()
    print("Options:")
    print("1. Scan for health value (enter current health)")
    print("2. Scan for stamina value (enter current stamina)")
    print("3. Get pattern from address (enter address in hex)")
    print("4. Exit")
    print()
    
    while True:
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            try:
                health = float(input("Enter current health value: "))
                print(f"Scanning for health = {health}...")
                addresses = scan_for_value(process, health, 'float')
                print(f"Found {len(addresses)} potential addresses")
                if addresses:
                    print("First 10 addresses:")
                    for addr in addresses[:10]:
                        pattern = get_pattern_from_address(process, addr)
                        print(f"  {hex(addr)}: {pattern}")
            except ValueError:
                print("Invalid input")
        
        elif choice == "2":
            try:
                stamina = float(input("Enter current stamina value: "))
                print(f"Scanning for stamina = {stamina}...")
                addresses = scan_for_value(process, stamina, 'float')
                print(f"Found {len(addresses)} potential addresses")
                if addresses:
                    print("First 10 addresses:")
                    for addr in addresses[:10]:
                        pattern = get_pattern_from_address(process, addr)
                        print(f"  {hex(addr)}: {pattern}")
            except ValueError:
                print("Invalid input")
        
        elif choice == "3":
            try:
                addr_str = input("Enter address (hex, e.g., 0x12345678): ").strip()
                address = int(addr_str, 16)
                pattern = get_pattern_from_address(process, address)
                if pattern:
                    print(f"Pattern: {pattern}")
                    print()
                    print("To use wildcards, replace changing bytes with ??")
                    print("Example: 48 8B 05 ?? ?? ?? ?? 48")
            except ValueError:
                print("Invalid address format")
        
        elif choice == "4":
            break
        
        else:
            print("Invalid choice")
        
        print()

if __name__ == "__main__":
    main()




