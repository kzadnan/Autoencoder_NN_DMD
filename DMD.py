import numpy as np
import matplotlib.pyplot as plt

def create_timestep_matrix(file_path, output_file="timestep_matrix.txt"):
    """
    Reads atomic positions from a LAMMPS trajectory file and converts them into a matrix.
    
    Stores:
    - Atom ID, Type, x, y, z in rows
    - Timesteps as columns
    - Box bounds separately
    
    Returns:
    - matrix: NumPy array (1664×5, timesteps)
    - sorted_timesteps: List of timesteps
    - box_bounds: Dictionary of box bounds for each timestep
    - num_atoms: Number of atoms detected in the file
    """

    timestep_data = {}
    box_bounds = {}
    current_timestep = None
    current_positions = []
    num_atoms = None  # To store detected number of atoms

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Detect timestep
            if "ITEM: TIMESTEP" in line:
                if current_positions and current_timestep is not None:
                    # Store previous timestep data
                    timestep_data[current_timestep] = np.array(current_positions).flatten()
                    current_positions = []  # Reset for next timestep
                
                i += 1
                current_timestep = int(lines[i].strip())

            # Read number of atoms
            elif "ITEM: NUMBER OF ATOMS" in line:
                i += 1
                num_atoms = int(lines[i].strip())

            # Read box bounds
            elif "ITEM: BOX BOUNDS" in line:
                i += 1
                x_bounds = list(map(float, lines[i].strip().split()))
                i += 1
                y_bounds = list(map(float, lines[i].strip().split()))
                i += 1
                z_bounds = list(map(float, lines[i].strip().split()))
                box_bounds[current_timestep] = (x_bounds, y_bounds, z_bounds)

            # Read atomic positions
            elif "ITEM: ATOMS" in line:
                i += 1
                for _ in range(num_atoms):
                    values = list(map(float, lines[i].strip().split()))
                    current_positions.extend(values)  # Flatten atom data into one long row
                    i += 1
                continue  # Prevent extra increment

            i += 1

    # Store last timestep
    if current_positions and current_timestep is not None:
        timestep_data[current_timestep] = np.array(current_positions).flatten()

    # Ensure timesteps are ordered correctly
    sorted_timesteps = sorted(timestep_data.keys())

    # Convert to matrix (1664 × 5, timesteps)
    num_rows = num_atoms * 5  # Each atom has (ID, Type, x, y, z)
    num_cols = len(sorted_timesteps)  # Number of time snapshots

    matrix = np.zeros((num_rows, num_cols))
    for j, timestep in enumerate(sorted_timesteps):
        matrix[:, j] = timestep_data[timestep]

    print(f"Matrix shape: {matrix.shape} (Rows: {num_rows}, Columns: {num_cols})")
    return matrix, sorted_timesteps, box_bounds, num_atoms

# Load trajectory
matrix, sorted_timesteps, box_bounds, num_atoms = create_timestep_matrix("nvt.tj")

#%%

# SVD calculation
U, S, VT = np.linalg.svd(matrix, full_matrices=False)

plt.figure()
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.plot(S, "r*")
plt.grid(True)
plt.show()

rank_x = np.linalg.matrix_rank(matrix)
print(f"Rank of the data matrix is {rank_x}")

#%%

# Step 1: Construct X and X'
X1 = matrix[:,:-1]  # All columns except the last
X2 = matrix[:,1:]   # All columns except the first

# Step 2: Compute A using least squares (Moore-Penrose pseudoinverse)
A = X2 @ np.linalg.pinv(X1)

# Step 3: Compute eigenvalues and eigenvectors of A
eigvals, W = np.linalg.eig(A)  # A W = W Λ

#%%

def reconstruct_nvt_file(matrix, sorted_timesteps, box_bounds, num_atoms, output_file="nvt_new.tj", mode_cutoff=rank_x):
    """
    Reconstructs a new LAMMPS trajectory file with only dominant SVD/DMD modes.

    Ensures:
    - Correct number of atoms
    - Box bounds are preserved
    
    Parameters:
    - matrix: (1664×5, timesteps)
    - sorted_timesteps: Ordered timestep list
    - box_bounds: Dict of box dimensions per timestep
    - num_atoms: Number of atoms in the system
    - output_file: Name of output trajectory file
    - mode_cutoff: Number of dominant modes to retain
    """

    num_timesteps = matrix.shape[1]  # Number of time snapshots

    # Step 1: Compute SVD
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    # Step 2: Keep dominant modes
    S_filtered = np.zeros_like(S)
    S_filtered[:mode_cutoff] = S[:mode_cutoff]
    S_matrix = np.diag(S_filtered)

    # Step 3: Reconstruct matrix using dominant modes
    matrix_filtered = U @ S_matrix @ VT

    # Step 4: Write new LAMMPS trajectory file
    with open(output_file, "w") as f:
        for t_idx, timestep in enumerate(sorted_timesteps):
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{num_atoms}\n")
            
            # Retrieve box bounds
            x_bounds, y_bounds, z_bounds = box_bounds[timestep]
            f.write("ITEM: BOX BOUNDS pp pp ss\n")
            f.write(f"{x_bounds[0]} {x_bounds[1]}\n")
            f.write(f"{y_bounds[0]} {y_bounds[1]}\n")
            f.write(f"{z_bounds[0]} {z_bounds[1]}\n")
            
            f.write("ITEM: ATOMS id type xs ys zs\n")

            # Write atom data for the timestep
            for i in range(num_atoms):
                idx = i * 5  # Position in flattened array
                atom_id = int(matrix_filtered[idx, t_idx])  # ID
                atom_type = round(matrix_filtered[idx + 1, t_idx])  # Type
                # if atom_type<1:
                #     atom_type=int(np.ceil(atom_type))
                # elif atom_type>3:
                #     atom_type=int(np.floor(atom_type))
                # else:
                #     atom_type=int(atom_type)
                
                x, y, z = matrix_filtered[idx + 2, t_idx], matrix_filtered[idx + 3, t_idx], matrix_filtered[idx + 4, t_idx]  # Positions
                f.write(f"{atom_id} {atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Filtered trajectory saved as {output_file} with {mode_cutoff} dominant modes.")

# Example usage

#%%
percentages=np.array([1,2,5,10,20,50,60,80,90,100])
for i in (percentages):
    
    reconstruct_nvt_file(matrix, sorted_timesteps, box_bounds, num_atoms, output_file=f"nvt_new{i}.tj", mode_cutoff=int(i*rank_x/100))

#%%

plt.figure()
plt.plot(S, "r*")  # Regular plot
plt.yscale("log")  # Apply log scale only to y-axis
plt.xlabel("Index")
plt.ylabel("Singular Value (log scale)")
plt.grid(True, which="both")  # Grid for better visualization
plt.show()


#%% visualization of the space eigen modes
# for i in range(len(U)):
#     plt.ylim([-0.2,0.2])
#     plt.plot(U[:,i],"r*")
#     plt.show()
# #%%

# for i in range(len(VT)):
#     plt.ylim([-0.1,0.1])
#     plt.plot(VT[:,i],"b*")
#     plt.show()    
