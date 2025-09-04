import numpy as np
import signac
from flow import FlowProject
from flow import directives
import flow.environments
import json
import os

class Project(FlowProject):
    pass

@Project.label
def rendered(job):
    return job.isfile("render.png")

@Project.post(rendered)
@Project.operation
def render(job):
    import numpy as np
    import gsd.hoomd
    import fresnel
    from random import randint

    # Create an image
    with job:
        # Read the trajectory
        with gsd.hoomd.open(name="dump.gsd", mode="r") as gsd_file:
            snap = gsd_file[-1]

        box = snap.configuration.box

        # Colour particles by type
        N = snap.particles.N
        individual_particle_types = snap.particles.types
        particle_types = snap.particles.typeid
        colors = np.empty((N, 3))
        colors_by_type = []
        for i in range(len(individual_particle_types)):
            color = '#%06X' % randint(0, 0xFFFFFF)
            color = list(int(color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
            colors_by_type.append(color)

        radii = np.zeros((snap.particles.typeid.shape))
         #['A', 'B', 'C', 'AP1', 'AM', 'AP2', 'BP1', 'BM', 'BP2', 'CP1', 'CM', 'CP2']
        radii[np.isin(snap.particles.typeid, [4, 7, 10])] = 1.0
        radii[np.isin(snap.particles.typeid, [3,5,6,8,9,11])] = 0.3

        # Color by typeid
        for i in range(len(individual_particle_types)):
            colors[particle_types == i] = fresnel.color.linear(colors_by_type[i])

        # Set the scene
        scene = fresnel.Scene(
            camera=fresnel.camera.Orthographic(
                position=(100, 100, 100),
                look_at=(0, 0, 0),
                up=(0, 1, 0),
                height=100,
            )
        )

        # Spheres for every particle in the system
        geometry = fresnel.geometry.Sphere(scene, N=N, radius=radii)
        geometry.position[:] = snap.particles.position
        geometry.material = fresnel.material.Material(roughness=0.1)
        geometry.outline_width = 0.05

        # use color instead of material.color
        geometry.material.primitive_color_mix = 1.0
        geometry.color[:] = fresnel.color.linear(colors)

        # create box in fresnel
        fresnel.geometry.Box(scene, box, box_radius=.07)

        # Render the system
        scene.lights = fresnel.light.lightbox()
        # out = fresnel.pathtrace(scene, light_samples=10, w=1380, h=1380)

        # Save image to file
        out = fresnel.preview(scene, w=3600, h=2220)
        print(out[:].shape)
        print(out[:].dtype)

        import PIL
        image = PIL.Image.fromarray(out[:], mode='RGBA')
        image.save('render.png')


@Project.label
def plotted(job):
    return (job.isfile("plots/pressure.png") 
            and job.isfile("plots/potential_energy.png") 
            and job.isfile("plots/translational_kinetic_energy.png") 
            and job.isfile("plots/rotational_kinetic_energy.png"))

@Project.post(plotted)
@Project.operation
def plot_quantities(job):
    import matplotlib.pyplot as plt
    import h5py
    
    with job: 

        data = h5py.File(name='dump_log.h5', mode='r')
        timestep = data['hoomd-data/Simulation/timestep'][:]
        pressure = data['/hoomd-data/md/compute/ThermodynamicQuantities/pressure'][:]
        potential_energy = data['/hoomd-data/md/compute/ThermodynamicQuantities/potential_energy'][:]
        translational_kinetic_energy = data['/hoomd-data/md/compute/ThermodynamicQuantities/translational_kinetic_energy'][:]
        rotational_kinetic_energy = data['/hoomd-data/md/compute/ThermodynamicQuantities/rotational_kinetic_energy'][:]

        if os.path.exists("plots") == False: 
            os.mkdir("plots")
        
        plt.close('all')
        fig, ax = plt.subplots()

        # Pressure
        ax.plot(timestep, pressure)
        ax.grid()
        ax.set(xlabel="timestep", ylabel="pressure", title="Pressure")
        fig.savefig("plots/pressure.png")

        # Potential energy
        ax.clear()
        ax.plot(timestep, potential_energy)
        ax.grid()
        ax.set(xlabel="timestep", ylabel="potential energy", title="Potential Energy")
        fig.savefig("plots/potential_energy.png")

        # Translational kinetic energy
        ax.clear()
        ax.plot(timestep, translational_kinetic_energy)
        ax.grid()
        ax.set(xlabel="timestep", ylabel="translational kinetic energy", title="Translational Kinetic Energy")
        fig.savefig("plots/translational_kinetic_energy.png")

        # Rotational Kinetic Energy
        ax.clear()
        ax.plot(timestep, rotational_kinetic_energy)
        ax.grid()
        ax.set(xlabel="timestep", ylabel="rotational kinetic energy", title="Rotational Kinetic Energy")
        fig.savefig("plots/rotational_kinetic_energy.png")

        
@Project.label
def polymers_identified(job):
    return(job.isfile("polymers/raw_point.png")
            and job.isfile("polymers/polymers.png")
            and job.isfile("polymers/polymer_size_dist.png")
            and job.isfile("polymer_dist.json"))

# @Project.pre.after(plot_quantities)
@Project.post(polymers_identified)
@Project.operation
def identify_polymers(job):
    import matplotlib.pyplot as plt
    import gsd.hoomd
    import freud

    with job:

        plt.close('all')

        if os.path.exists(job.fn("polymers")) == False: 
            os.mkdir("polymers")

        # Read equilibrated system from GSD file
        with gsd.hoomd.open(name="dump.gsd", mode="r") as gsd_file:
            frames = gsd_file[-10:]
            polymer_sizes_by_frame = {}
            for i, frame in enumerate(frames):
                current_frame = int(frame.configuration.step / 1e5)
                positions = []
                for index in range(frame.particles.N):
                    if frame.particles.typeid[index] == 0 or frame.particles.typeid[index] == 1 or frame.particles.typeid[index] == 2:
                        positions.append(frame.particles.position[index])
                
                box = freud.box.Box.from_box(frame.configuration.box, dimensions = 3)
                system = freud.AABBQuery(box, np.array(positions))

                # Plot and save central particle positions before clustering. 
                if i == len(frames) - 1:
                    plt.clf()
                    fig = plt.figure()
                    system.plot(ax=fig.add_subplot(projection='3d'), s=10)
                    # plt.title('Raw points before clustering', fontsize=20)
                    plt.gca().tick_params(axis='both', which='both', labelsize=7, size=4)
                    plt.savefig("polymers/raw_point.png")

                # Identify polymers
                cl = freud.cluster.Cluster()
                cl.compute(system, neighbors={"mode": "ball", "r_max": 2.1})
                print(cl.cluster_idx)
                print(cl.num_clusters)

                # Plot polymers
                if i == len(frames) - 1:
                    plt.clf()
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    for cluster_id in range(cl.num_clusters):
                        cluster_system = freud.AABBQuery(system.box, system.points[cl.cluster_keys[cluster_id]])
                        cluster_system.plot(ax=ax, s=10, label="Cluster {}".format(cluster_id))
                        # print("There are {} points in cluster {}.".format(len(cl.cluster_keys[cluster_id]), cluster_id))

                    # ax.set_title('Clusters identified', fontsize=20)
                    ax.legend(loc='right', fontsize=4)
                    ax.tick_params(axis='both', which='both', labelsize=7, size=4)
                    plt.savefig("polymers/polymers.png")

                # Calculate occurences of polymer sizes
                lengths = []
                polymers = []

                for cluster_id in range(cl.num_clusters):
                    if len(cl.cluster_keys[cluster_id]) > len(lengths):
                        lengths.extend(
                            [0] * (len(cl.cluster_keys[cluster_id]) - len(lengths))
                            )
                    lengths[len(cl.cluster_keys[cluster_id]) - 1] += 1
                    
                    polymers += [len(cl.cluster_keys[cluster_id])]

                print(lengths)
                print(polymers)

                polymer_sizes = {}
                for length in range(len(lengths)):
                    polymer_sizes[f"{str(length + 1)} panels"] = lengths[length]
                polymer_sizes_by_frame[f"Frame {int(frame.configuration.step / 1e5)}"] = polymer_sizes

            # Save polymer size distribution in a json file
            polymer_size_averages = {}
            for key in polymer_sizes_by_frame:
                for subkey in polymer_sizes_by_frame[key]:
                    if subkey in polymer_size_averages.keys():
                        polymer_size_averages[subkey] += np.array([polymer_sizes_by_frame[key][subkey],1])
                    else: 
                        polymer_size_averages[subkey] = np.array([polymer_sizes_by_frame[key][subkey],1])
            for key in polymer_size_averages:
                polymer_size_averages[key] = round(polymer_size_averages[key][0]/polymer_size_averages[key][1])
            polymer_sizes_by_frame["Averages"] = dict(sorted(polymer_size_averages.items(), key=lambda item: int(item[0].split(" ")[0])))

            with open("polymer_dist.json", "w") as json_file:
                json.dump(polymer_sizes_by_frame, json_file, indent=4, sort_keys=True)

            # Plot polymer size distribution (procedure from https://matplotlib.org/stable/gallery/statistics/barchart_demo.html)
            plt.clf()
            fig, ax1 = plt.subplots(figsize=(9, 7))  # Create the figure
            fig.subplots_adjust(left=0.215, right=0.88)
            fig.canvas.manager.set_window_title('Polymer Size Distribution')

            pos = np.array(np.arange(len(polymer_size_averages))) + 1

            rects = ax1.barh(pos, [polymer_size_averages[key] for key in sorted(polymer_size_averages, key=lambda item: float(item.split(" ")[0]))],
                                align='center',
                                height=0.5,
                                tick_label=[str(p) + " Monomer" + ("s" if p != 1 else "") for p in pos])

            ax1.set_title(f"Polymer Size Distribution")
            ax1.set_ylabel("Polymer Size")

            ax1.set_xlim([0, 100])
            ax1.xaxis.grid(True, linestyle='--', which='major',
                            color='grey', alpha=.25)

            # Write the sizes inside each bar to aid in interpretation
            rect_labels = []
            for rect in rects:
                width = rect.get_width()
                # The bars aren't wide enough to print the ranking inside
                if width < 40:
                    # Shift the text to the right side of the right edge
                    xloc = 5
                    # Black against white background
                    clr = 'black'
                    align = 'left'
                else:
                    # Shift the text to the left side of the right edge
                    xloc = -5
                    # White on magenta
                    clr = 'white'
                    align = 'right'

                # Center the text vertically in the bar
                yloc = rect.get_y() + rect.get_height() / 2
                label = ax1.annotate(
                    width, xy=(width, yloc), xytext=(xloc, 0),
                    textcoords="offset points",
                    horizontalalignment=align, verticalalignment='center',
                    color=clr, weight='bold', clip_on=True)
                rect_labels.append(label)

            plt.savefig("polymers/polymer_size_dist.png")


@Project.label
def target_cluster_yield_identified(job):
    return job.isfile("target_cluster_yield.json") and job.isfile("clusters.json")

@Project.post(target_cluster_yield_identified)
@Project.operation
def target_cluster_yield(job):
    import gsd.hoomd
    import freud
    from scipy.spatial.transform import Rotation


    # Now, we want to determine whether two clusters are of the same type.
    class Cluster():
        def __init__(self,monomer_type_order,monomer_headings,circular=False):
            self.monomer_type_order = monomer_type_order
            self.monomer_headings = monomer_headings
            self.circular = circular
            assert(len(self.monomer_type_order) == len(self.monomer_headings))
        
        @property
        def length(self):
            return len(self.monomer_type_order)

        def compare(self, cluster2):
            if self.length == cluster2.length:
                # the two clusters could be of the same type.
                if self.length == 1:
                    if self.monomer_type_order == cluster2.monomer_type_order:
                        return True
                elif not self.circular and not cluster2.circular:
                    end_monomers1 = [self.monomer_type_order[0],self.monomer_type_order[-1]]
                    if end_monomers1 == [cluster2.monomer_type_order[0],cluster2.monomer_type_order[-1]]:
                        cluster2_order = cluster2.monomer_type_order
                        cluster2_headings = cluster2.monomer_headings
                    elif end_monomers1 == [cluster2.monomer_type_order[-1],cluster2.monomer_type_order[0]]:
                        cluster2_order = cluster2.monomer_type_order
                        cluster2_order = cluster2_order[::-1]
                        cluster2_headings = cluster2.monomer_headings
                        cluster2_headings.reverse()
                    else:
                        return False
                    if self.monomer_type_order == cluster2_order and self.monomer_headings == cluster2_headings:
                        return True
                elif self.circular and cluster2.circular:
                    # dealing with a circular polymer, which means that matching will take a bit more work.
                    cluster2_order = cluster2.monomer_type_order
                    cluster2_headings = cluster2.monomer_headings
                    double_cluster2_order = cluster2_order + cluster2_order
                    cyclic_index = double_cluster2_order.find(self.monomer_type_order)
                    if cyclic_index != -1:
                        double_cluster2_headings = cluster2_headings + cluster2_headings
                        if self.monomer_headings == double_cluster2_headings[cyclic_index:cyclic_index+len(cluster2_headings)]:
                            return True
            return False


    target_cluster = Cluster("ABC",[1,1,1],False)
    cluster_search_radius = 2.1

    with job:

        # Read equilibrated system from GSD file
        with gsd.hoomd.open(name="dump.gsd", mode="r") as gsd_file:

            frames = gsd_file[-10:] 
            target_cluster_yields = []

            def extend_cluster_order(cluster_nbrs, cluster_order, current_monomer):
                for bond in cluster_nbrs[current_monomer]:
                    if bond[1] not in cluster_order:
                        cluster_order.append(bond[1])
                        break
                current_monomer = cluster_order[-1]
                return cluster_order, current_monomer
            
            all_clusters = []

            for i, frame in enumerate(frames): 
                current_frame = int(frame.configuration.step / 1e5)
                positions = []
                for index in range(frame.particles.N):
                    if frame.particles.typeid[index] == 0 or frame.particles.typeid[index] == 1 or frame.particles.typeid[index] == 2:
                        positions.append(frame.particles.position[index])

                box = freud.box.Box.from_box(frame.configuration.box, dimensions = 3)
                system = freud.AABBQuery(box, np.array(positions))

                # Identify polymers
                cl = freud.cluster.Cluster()
                cl.compute(system, neighbors={"mode": "ball", "r_max": cluster_search_radius})

                typeid_to_type = ['A','B','C']

                # Find order of monomers in a cluster.
                # cluster_type_orders = []
                # cluster_headings = []
                # cluster_circ = []
                clusters = []

                for cluster in cl.cluster_keys:
                    cluster_nbrs = {}
                    cluster_order = []
                    circular = False
                    for monomer_idx in cluster:
                        immediate_neighbors =[]
                        # Find up to two nearby neighbors
                        for bond in system.query(system.points, dict(num_neighbors=2, r_max=cluster_search_radius, exclude_ii=True)):
                            # Make sure that particles are not too close together (probably more of an issue when using the repulsive potential)
                            if bond[2] < 1.5:
                                print(f"[Warning] Short bond between particles {monomer_idx} and {bond[1]} (length: {bond[2]})")
                            if bond[0] == monomer_idx and bond[1] in cluster:
                                immediate_neighbors.append(bond) #immediate_neighbors.append(bond[1])
                        cluster_nbrs[monomer_idx] = immediate_neighbors
                        # Order wrt bond length?

                    single_neighbor_monomers = []
                    # Find the monomers at the end of a chain and proceed by computing the order in which the monomers in the cluster are bonded to each other
                    for monomer_index in cluster_nbrs:
                        if len(cluster_nbrs[monomer_index]) == 1:
                            single_neighbor_monomers.append(monomer_index)
                    if len(cluster) > 1:
                        try:
                            assert(len(single_neighbor_monomers) == 2)
                        except:
                            if len(single_neighbor_monomers) == 0:
                                print("No single-neighbor monomers found.  Checking for likely circular monomer...")
                                try:
                                    neighbors = set({})
                                    for monomer_idx in cluster:
                                        assert(len(cluster_nbrs[monomer_idx]) == 2)
                                        for bond in cluster_nbrs[monomer_idx]:
                                            neighbors.add(bond[1])
                                    assert(set(cluster) == set(neighbors))
                                except:
                                    raise AssertionError("No single-neighbor monomers found, and cluster does not seem to be circular.")
                                print("Circular polymer likely found.")
                                circular = True
                            else:
                                raise AssertionError(f"there are {len(single_neighbor_monomers)} monomers with single neighbors, but there should be 0 or 2.")

                        if not circular:
                            cluster_order = [single_neighbor_monomers[0]]
                            current_monomer = single_neighbor_monomers[0]
                            while single_neighbor_monomers[1] not in cluster_order:
                                cluster_order, current_monomer = extend_cluster_order(cluster_nbrs, cluster_order, current_monomer)
                        else: # Dealing with a circular polymer
                            cluster_order = [list(cluster_nbrs.keys())[0]]
                            current_monomer = cluster_order[-1]
                            while set(cluster) != set(cluster_order):
                                cluster_order, current_monomer = extend_cluster_order(cluster_nbrs, cluster_order, current_monomer)

                    else:
                        assert(len(single_neighbor_monomers) == 0)
                        cluster_order = cluster

                    assert(set(cluster) == set(cluster_order))

                    headings = []
                    if len(cluster) > 1:
                        # Determine which side of the monomer a bonded neighbor is on.  In this case, we are particularly interested in finding out the orientation of our chain.
                        monomer1_idx = cluster_order[0]
                        monomer2_idx = cluster_order[1]

                        # in this case, any monomer can be forward or backwards, so we need to take all these cases into account.
                        monomer1_pos = positions[monomer1_idx]
                        monomer2_pos = positions[monomer2_idx]
                        monomer1_or = frame.particles.orientation[monomer1_idx]
                        monomer2_or = frame.particles.orientation[monomer2_idx]
                        monomer1_dir = Rotation.from_quat(monomer1_or, scalar_first=True).apply([1.0,0.0,0.0])
                        monomer2_dir = Rotation.from_quat(monomer2_or, scalar_first=True).apply([1.0,0.0,0.0])
                        dist = np.linalg.norm(monomer2_pos - monomer1_pos)
                        # assert(np.dot(monomer1_dir,monomer2_dir) > 0)
                        if np.linalg.norm(monomer1_pos + monomer1_dir * dist - monomer2_pos) < np.linalg.norm(monomer1_pos - monomer1_dir * dist - monomer2_pos):
                            # Monomer 1 in direction of monomer 2
                            headings.append(1)
                        else:
                            # Monomer 1 points away from monomer 2
                            if not circular:
                                headings.append(-1)
                            else:
                                cluster_order.reverse()
                                cluster_order = [monomer1_idx] + cluster_order[:-1]
                                headings.append(1)
                        for i in range(len(cluster_order)-1):
                            monomer1_idx = cluster_order[i]
                            monomer2_idx = cluster_order[i+1]
                            monomer1_or = frame.particles.orientation[monomer1_idx]
                            monomer2_or = frame.particles.orientation[monomer2_idx]
                            monomer1_dir = Rotation.from_quat(monomer1_or, scalar_first=True).apply([1.0,0.0,0.0])
                            monomer2_dir = Rotation.from_quat(monomer2_or, scalar_first=True).apply([1.0,0.0,0.0])
                            if headings[0]*np.dot(monomer1_dir,monomer2_dir) > 0:
                                headings.append(1)
                            else:
                                headings.append(-1)
                    else:
                        headings.append(1)

                    # From here, we rewrite the cluster composition in a more readable format.
                    cluster_type_order = "".join(typeid_to_type[id] for id in frame.particles.typeid[cluster_order])
                    print(cluster_type_order)
                    # cluster_type_orders.append(cluster_type_order)
                    # cluster_headings.append(headings)
                    # cluster_circ.append(circular)
                    cluster_details = Cluster(cluster_type_order, headings, circular)
                    clusters.append(cluster_details)
                
                all_clusters.extend(clusters)

                num_abc_polymers = 0
                
                for ind_cluster in clusters:
                    if target_cluster.compare(ind_cluster):
                        num_abc_polymers += 1
                
                target_cluster_yield = num_abc_polymers/len(cl.cluster_keys)
                target_cluster_yields.append(target_cluster_yield)

            avg_target_cluster_yield = np.mean(target_cluster_yields)
        # with open("polymer_order.json", "w") as json_file:
        #     json.dump(cluster_type_orders, json_file, indent=4)

        # with open("polymer_headings.json", "w") as json_file:
        #     json.dump(cluster_headings, json_file, indent=4)
        
        with open("target_cluster_yield.json", "w") as json_file:
            json.dump([avg_target_cluster_yield], json_file)
        
        # Only keep unique cluster in all_clusters
        same = np.zeros((len(all_clusters),len(all_clusters)))
        for i in range(len(all_clusters)):
            for j in range(len(all_clusters)):
                same[i,j] = all_clusters[i].compare(all_clusters[j])
        
        indices_to_remove = set({})
        for i in range(len(all_clusters)):
            for j in range(len(all_clusters)):
                if j > i:
                    if same[i,j]:
                        indices_to_remove.add(j)
        
        indices_to_remove = sorted(indices_to_remove, reverse=True)

        for index in indices_to_remove:
            all_clusters.pop(index)
        
        all_clusters = [vars(cluster) for cluster in all_clusters]

        with open("clusters.json", "w") as json_file:
            json.dump(all_clusters, json_file)

if __name__ == '__main__':
    Project().main()
