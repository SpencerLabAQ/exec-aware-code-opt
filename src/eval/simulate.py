import sys
import m5
from m5.objects import *
import argparse

def create_standard_system():
      
      '''
      The System object contains a lot of functional (not timing-level) information, 
      like the physical memory ranges, the root clock domain, the root voltage domain, 
      the kernel (in full-system simulation), etc. 
      To create the system SimObject, we simply instantiate it like a normal python class
      '''
      system = System()

      # Create clock
      system.clk_domain = SrcClockDomain()
      system.clk_domain.clock = '3.5GHz'
      system.clk_domain.voltage_domain = VoltageDomain()

      '''
      let’s set up how the memory will be simulated. We are going to use timing mode for the memory simulation
      '''
      system.mem_mode = 'timing'
      system.mem_ranges = [AddrRange('8192MB')]
      # Memory boost
      # system.mem_mode = 'timing'
      # mem_size = '32GB'
      # system.mem_ranges = [AddrRange('100MB'), # For kernel
      #                   AddrRange(0xC0000000, size=0x100000), # For I/0
      #                   AddrRange(Addr('4GB'), size = mem_size) # All data
      #                         ]

      # We’ll start with the most simple timing-based CPU in gem5 for the X86 ISA, X86TimingSimpleCPU
      system.cpu = X86TimingSimpleCPU()

      # create the system-wide memory bus
      system.membus = SystemXBar()

      # connect the cache ports on the CPU to it
      system.cpu.icache_port = system.membus.cpu_side_ports
      system.cpu.dcache_port = system.membus.cpu_side_ports

      '''
      Connecting the PIO and interrupt ports to the memory bus is an x86-specific requirement. 
      Other ISAs (e.g., ARM) do not require these 3 extra lines.
      '''
      system.cpu.createInterruptController()
      system.cpu.interrupts[0].pio = system.membus.mem_side_ports
      system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
      system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

      system.system_port = system.membus.cpu_side_ports

      # Memory controller
      system.mem_ctrl = MemCtrl()
      system.mem_ctrl.dram = DDR3_1600_8x8()
      system.mem_ctrl.dram.range = system.mem_ranges[0]
      system.mem_ctrl.port = system.membus.mem_side_ports

      '''
      Full system vs syscall emulation

      gem5 can run in two different modes called “syscall emulation” and “full system” or SE and FS modes. 
      In full system mode (covered later full-system-part), gem5 emulates the entire hardware system and runs an unmodified kernel. 
      Full system mode is similar to running a virtual machine.

      Syscall emulation mode, on the other hand, does not emulate all of the devices in a system and focuses 
      on simulating the CPU and memory system. Syscall emulation is much easier to configure since you are 
      not required to instantiate all of the hardware devices required in a real system. 
      However, syscall emulation only emulates Linux system calls, and thus only models user-mode code.

      If you do not need to model the operating system for your research questions, 
      and you want extra performance, you should use SE mode. 
      However, if you need high fidelity modeling of the system, 
      or OS interaction like page table walks are important, then you should use FS mode.
      '''

      return system

def create_skylake_config():

      _CPUModel = BaseCPU

      system = System()
      
      system.clk_domain = SrcClockDomain()
      system.clk_domain.clock = '3.5GHz'
      system.clk_domain.voltage_domain = VoltageDomain()

      system.mem_mode = 'timing'
      mem_size = '32GB'
      system.mem_ranges = [AddrRange('100MB'), # For kernel
                        AddrRange(0xC0000000, size=0x100000), # For I/0
                        AddrRange(Addr('4GB'), size = mem_size) # All data
                              ]

      system.cpu = _CPUModel

      # Create a memory bus
      system.membus = SystemXBar(width = 192)
      system.membus.badaddr_responder = BadAddr()
      system.membus.default = system.badaddr_responder.pio

      # Set up the system port for functional access from the simulator
      system.system_port = system.membus.cpu_side_ports

      # Create an L1 instruction and data cache
      system.cpu.icache = L1ICache()
      system.cpu.dcache = L1DCache()
      system.cpu.mmucache = MMUCache()

      # Connect the instruction and data caches to the CPU
      system.cpu.icache.connectCPU(system.cpu)
      system.cpu.dcache.connectCPU(system.cpu)
      system.cpu.mmucache.connectCPU(system.cpu)

      # Create a memory bus, a coherent crossbar, in this case
      system.l2bus = L2XBar(width = 192)

      # Hook the CPU ports up to the l2bus
      system.cpu.icache.connectBus(system.l2bus)
      system.cpu.dcache.connectBus(system.l2bus)
      system.cpu.mmucache.connectBus(system.l2bus)

      # Create an L2 cache and connect it to the l2bus
      system.l2cache = L2Cache()
      system.l2cache.connectCPUSideBus(system.l2bus)

      # Create a memory bus, a coherent crossbar, in this case
      system.l3bus = L2XBar(width = 192,
                              snoop_filter = SnoopFilter(max_capacity='32MB'))

      # Connect the L2 cache to the l3bus
      system.l2cache.connectMemSideBus(system.l3bus)

      # Create an L3 cache and connect it to the l3bus
      system.l3cache = L3Cache()
      system.l3cache.connectCPUSideBus(system.l3bus)

      # Connect the L3 cache to the membus
      system.l3cache.connectMemSideBus(system.membus)

      # create the interrupt controller for the CPU
      system.cpu.createInterruptController()

      system.cpu.interrupts[0].pio = system.membus.mem_side_ports
      system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
      system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

      system.createMemoryControllersDDR4()

      # provide cache paramters for verbatim CPU
      if (_CPUModel is VerbatimCPU):
            # L1I-Cache
            system.cpu.icache.size = '32kB'
            system.cpu.icache.tag_latency = 4
            system.cpu.icache.data_latency = 4
            system.cpu.icache.response_latency = 1
            # L1D-Cache
            system.cpu.dcache.tag_latency = 4
            system.cpu.dcache.data_latency = 4
            system.cpu.dcache.response_latency = 1

      return system
    
# system = create_skylake_config()
system = create_standard_system()
##########
# NEW
##########
parser = argparse.ArgumentParser()
parser.add_argument('exec_file')
parser.add_argument('stdin_file')
args = parser.parse_args()

binary = args.exec_file
stdin_file = args.stdin_file

# print(f"Simulating.... {binary=} {stdin_file=}")

# Run the experiment in SE mode

# for gem5 V21 and beyond
system.workload = SEWorkload.init_compatible(binary)

process = Process()
process.cmd = [binary]
process.input = stdin_file
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system = False, system = system)
m5.instantiate()

# print("Beginning simulation!")
exit_event = m5.simulate()

# print('Exiting @ tick {} because {}'
#       .format(m5.curTick(), exit_event.getCause()))