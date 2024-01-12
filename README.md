# Quantum circuit simulation with Qiskit and Python

The Qiskit package provides the basic building blocks necessary to program quantum computers.
A basic workflow using Qiskit consists of two stages: Build and Execute. 
That is, one can make different quantum circuits that represent the problem to solve (build phase), and then run them as a job on backends (execute phase).
To simulate quantum circuit locally, the ``qasm_simulator`` backend can be chosen, which will run the Qiskit AerSimulator that interface many different simulation methods depending of the circuit and the gate used.
Aer simulator documentation is available on [Qiskit website](https://qiskit.org/ecosystem/aer/stubs/qiskit_aer.AerSimulator.html).


## Installation

Building quantum circuit is done with the ``qiskit`` package, while quantum circuit simulation requires the ``qiskit-aer`` package.
The latest qiskit (0.44.2) and qiskit-aer version (0.12.2) is available from Python 3.9 to 3.11.
It can be installed in a virtual environment with (using Python 3.9 for example)
```
module load python/3.9 scipy-stack/2023b symengine/0.9.0
virtualenv --no-download env_qiskit
source env_qiskit/bin/activate
pip install qiskit==0.44.2 qiskit-aer==0.12.2 --no-index
deactivate
```

In the job submission script, the environment must be sourced as
```
module load scipy-stack/2023b symengine/0.9.0
source env_qiskit/bin/activate
```

Qiskit-aer wheel in version 0.12.2 is build with FlexiBLAS abstraction layer for BLAS operations (see [BLAS and LAPACK]
()), MPI for multi-nodes, CUDA for single-node GPU simulations, and CUDA libraries [cuTENSOR](https://developer.nvidia.com/cutensor) and [cuQuantum](https://developer.nvidia.com/cuquantum-sdk) for faster GPU-based state-vector and tensor-network simulations.
GPU simulations can be run on device with Compute Capability from 7.0 to 8.6 including NVIDIA A40 GPU on IQ cluster.


## Example

### Code

The following generates a random quantum circuit that is simulated with AerSimulator (the simulator in qiskit-aer package):

```
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import Aer

# Create a random circuit
n_qubits = 28
depth = 30
qc = random_circuit(n_qubits, depth, measure=True, seed=1234)

# Execute it on local AerSimulator
measured_qc = qc.measure_all(inplace=False)
backend = Aer.get_backend('qasm_simulator')
backend_options = {
    "method":"statevector",
    "device":"CPU"
}
result = backend.run(transpile(measured_qc, backend), **backend_options, shots=1024).result()
counts  = result.get_counts(measured_qc)
print(counts)
```


### Single-node simulation with OpenMP

The previous random circuit can be run as is on a single-node with OpenMP parallelization with 5GB of memory and 8 CPUs:

```
#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load scipy-stack/2023b symengine/0.9.0
source env_qiskit/bin/activate
python my_circuit.py
```

Runtime on Narval with 8 CPUs: 339 seconds.


### GPU simulation

The random circuit can be run on the GPU by modifying the backend options:
```
backend_options = {
    "method":"statevector",
    "device":"GPU",
    "cuStateVec_enable":False,
```

and the job submission script to select a GPU (See [[Using multiple GPUs with SLURM]]):
```
#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem-per-cpu=1G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load scipy-stack/2023b symengine/0.9.0
source env_qiskit/bin/activate
python my_circuit.py
```

Runtime on Narval A100 GPUs: 8 seconds!

GPU simulation with the statevector method is fast and can be more than two order of magnitude quicker than the CPU version.
Faster statevector simulations could also be obtained by using the CuStateVec library from NVIDIA with the ``cuStateVec_enable`` option set to ``True``.
For example, increasing the size of the circuit from 28 to 31 qubits and the depth from 30 to 60 gates, the runtime on A100 GPU goes from 43 seconds without CuStateVec to 34 seconds with CuStateVec.

However, statevector method memory requirement scale as 2**(n_qubit+4), i.e. adding one more qubit double the memory requirement.
As an example, a 28 qubits circuit requires less than 4 GB of memory, while a 32 qubits circuit requires approximately 70 GB of memory.
Therefore, GPU memory limits the size of the quantum circuit that can be simulated (up to 31 qubits on A100 40GB GPU).
Note that the Aer simulator won't complain if not enough memory is available on the GPUs, but the job will hangs until the end of the time allowed.
To overcome this limitation, user might choose the other available simulation methods such as the tensor network method available with NVIDIA cuTENSOR library:
```
backend_options = {
    "method":"tensor_network",
    "device":"GPU",
}
```

Tensor network methods limits the memory needed to run the calculation but runtime can be significantly longer depending on the depth of the circuit.
