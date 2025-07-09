import math
import matplotlib.pyplot as plt
import networkx as nx

class Task:
    def __init__(self, id, size):
        self.id = id
        self.size = size  # Task runtime in seconds (ref_time)
        self.predecessors = []
        self.successors = []

class VM:
    def __init__(self, type_id, processing_capacity, cost_per_interval, bandwidth):
        self.type_id = type_id
        self.processing_capacity = processing_capacity  # Compute Units (CU)
        self.cost_per_interval = cost_per_interval  # Cost per hour
        self.bandwidth = bandwidth  # MB/s
        self.tasks = []  # Tasks assigned to this VM
        self.lease_start_time = 0
        self.lease_finish_time = 0

def parse_dag_file(dag_file_path):
    """
    Parse DAG file to create tasks and edges dictionaries.
    Args:
        dag_file_path: Path to the DAG file.
    Returns:
        Tuple (tasks, edges) where tasks is a dictionary of Task objects and edges is a dictionary of edge weights.
    """
    tasks = {}
    edges = {}
    file_sizes = {}

    with open(dag_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('FILE'):
            parts = line.strip().split()
            file_name = parts[1]
            file_size = int(parts[2]) / (1024 * 1024)  # Convert bytes to MB
            file_sizes[file_name] = round(file_size, 2)

    for line in lines:
        if line.startswith('TASK'):
            parts = line.strip().split()
            task_id = parts[1]
            runtime = float(parts[3])  # Runtime in seconds
            tasks[task_id] = Task(task_id, size=runtime)
    
    task_inputs = {}
    task_outputs = {}
    for line in lines:
        if line.startswith('INPUTS'):
            parts = line.strip().split()
            task_id = parts[1]
            input_files = parts[2:] if len(parts) > 2 else []
            task_inputs[task_id] = input_files
        if line.startswith('OUTPUTS'):
            parts = line.strip().split()
            task_id = parts[1]
            output_files = parts[2:] if len(parts) > 2 else []
            task_outputs[task_id] = output_files
    
    for line in lines:
        if line.startswith('EDGE'):
            parts = line.strip().split()
            parent_id = parts[1]
            child_id = parts[2]
            if parent_id in tasks and child_id in tasks:
                if child_id not in edges:
                    edges[child_id] = {}
                data_size = next((file_sizes.get(f, 0) for f in task_outputs.get(parent_id, []) if f in task_inputs.get(child_id, [])), 0)
                edges[child_id][parent_id] = data_size
                tasks[child_id].predecessors.append(parent_id)
                tasks[parent_id].successors.append(child_id)
    
    entry_tasks = [tid for tid in tasks if not tasks[tid].predecessors]
    exit_tasks = [tid for tid in tasks if not tasks[tid].successors]
    
    if len(entry_tasks) > 1:
        tasks['t_en'] = Task('t_en', size=0)
        for et in entry_tasks:
            if 't_en' not in edges:
                edges['t_en'] = {}
            edges[et] = {'t_en': 0}
            tasks[et].predecessors.append('t_en')
            tasks['t_en'].successors.append(et)
    
    if len(exit_tasks) > 1:
        tasks['t_ex'] = Task('t_ex', size=0)
        for et in exit_tasks:
            if 't_ex' not in edges:
                edges['t_ex'] = {}
            edges['t_ex'][et] = 0
            tasks['t_ex'].predecessors.append(et)
            tasks[et].successors.append('t_ex')
    
    return tasks, edges

def calculate_efficiency(vm):
    return vm.processing_capacity / vm.cost_per_interval

def greedy_resource_provisioning(resource_pool, budget, T=3600):
    """
    Greedy Resource Provisioning (Section 4.1.1).
    Args:
        resource_pool: List of VM objects.
        budget: Budget in dollars.
        T: Billing interval in seconds (default 3600).
    Returns:
        List of VM instances (type_id, instance_id).
    """
    for vm in resource_pool:
        vm.efficiency = calculate_efficiency(vm)
    sorted_pool = sorted(resource_pool, key=lambda x: x.efficiency, reverse=True)
    
    M = []
    remaining_budget = budget
    instance_id = 0
    
    for vm_type in sorted_pool:
        cost_per_hour = vm_type.cost_per_interval
        initial_hours = math.ceil(max(vm_type.lease_finish_time, 1e-10) / T) if vm_type.lease_finish_time > 0 else 0
        initial_cost = initial_hours * cost_per_hour
        if initial_cost > remaining_budget:
            continue
        max_instances = math.floor((remaining_budget - initial_cost) / cost_per_hour)
        for _ in range(min(max_instances, 5)):
            M.append((vm_type.type_id, instance_id))
            remaining_budget -= cost_per_hour
            instance_id += 1
        if remaining_budget <= 0:
            break
    
    return M

def compute_rank_u(tasks, edges, resource_pool, bandwidth):
    """
    Compute upward rank for task prioritization (Section 4.1.2).
    """
    avg_processing_capacity = sum(vm.processing_capacity for vm in resource_pool) / len(resource_pool)
    avg_comm_cost = 1 / bandwidth
    
    ranku = {}
    def calc_ranku(task_id):
        if task_id in ranku or task_id in ['t_en', 't_ex']:
            return ranku.get(task_id, 0)
        
        if task_id not in tasks:
            return 0
        
        task = tasks[task_id]
        w_i = task.size / avg_processing_capacity
        max_succ_cost = 0
        for succ_id in task.successors:
            comm_cost = edges.get(succ_id, {}).get(task_id, 0) * avg_comm_cost
            succ_rank = calc_ranku(succ_id)
            max_succ_cost = max(max_succ_cost, comm_cost + succ_rank)
        
        ranku[task_id] = w_i + max_succ_cost
        return ranku[task_id]
    
    for task_id in tasks:
        calc_ranku(task_id)
    
    tasks_list = [(tid, val) for tid, val in ranku.items() if tid not in ['t_en', 't_ex']]
    tasks_list.sort(key=lambda x: (-x[1], x[0]))
    return {tid: i for i, (tid, _) in enumerate(tasks_list)}, tasks_list

def modified_heft(tasks, edges, M, resource_pool, budget, bandwidth=20, T=3600):
    """
    Modified HEFT with Budget-Violation Avoidance (Section 4.1.2).
    Args:
        tasks: Dictionary of Task objects.
        edges: Dictionary of edge weights (data sizes in MB).
        M: List of (vm_type, instance_id) tuples.
        resource_pool: List of VM types.
        budget: Budget in dollars.
        bandwidth: Communication bandwidth in MB/s.
        T: Billing interval in seconds.
    Returns:
        Tuple (M, AL, Z) where AL is [(task_id, (vm_type, instance_id), ST, FT)] and Z is task order.
    """
    AL = []
    vm_usage = {vm_id: next((vm.lease_finish_time for vm in resource_pool if vm.type_id == vm_id[0]), 0) for vm_id in M}
    P0, sorted_tasks = compute_rank_u(tasks, edges, resource_pool, bandwidth)
    sorted_task_ids = [tid for tid, _ in sorted_tasks]

    def compute_EFT(task_id, vm_type, instance_id):
        task = tasks[task_id]
        vm_id = (vm_type, instance_id)
        est = 0
        preds = task.predecessors
        if preds:
            max_parent_ft = 0
            for parent in preds:
                parent_ft = next((ft for t_id, _, _, ft in AL if t_id == parent), 0)
                parent_vm_type = next((v[0] for t_id, v, _, _ in AL if t_id == parent), None)
                comm_cost = edges.get(task_id, {}).get(parent, 0) / bandwidth if parent_vm_type and vm_type != parent_vm_type else 0
                max_parent_ft = max(max_parent_ft, parent_ft + comm_cost)
            est = max_parent_ft
        
        et = task.size / next(v.processing_capacity for v in resource_pool if v.type_id == vm_type)
        st = max(est, vm_usage[vm_id])
        ft = st + et
        return st, ft, et

    def compute_cost(temp_usage, used_vms):
        total_cost = 0
        for vm_id in M:
            vm_type = vm_id[0]
            cost_per_hour = next(v.cost_per_interval for v in resource_pool if v.type_id == vm_type)
            usage = temp_usage[vm_id]
            if vm_id in used_vms or usage > 0:
                total_cost += cost_per_hour * math.ceil(max(usage, 1e-10) / T)
        return total_cost

    used_vms = set()
    for task_id in sorted_task_ids:
        best_eft = float('inf')
        best_vm = None
        best_st = 0
        best_cost = float('inf')
        best_et = 0

        for vm_type, instance_id in sorted(M, key=lambda x: next(v.processing_capacity for v in resource_pool if v.type_id == x[0]), reverse=True):
            st, eft, et = compute_EFT(task_id, vm_type, instance_id)
            temp_usage = vm_usage.copy()
            temp_usage[(vm_type, instance_id)] = max(temp_usage[(vm_type, instance_id)], eft)
            temp_used_vms = used_vms | {(vm_type, instance_id)}
            cost = compute_cost(temp_usage, temp_used_vms)
            
            if cost <= budget + 1e-10 and eft < best_eft:
                best_eft = eft
                best_vm = (vm_type, instance_id)
                best_st = st
                best_cost = cost
                best_et = et

        if best_vm is None:
            for vm_type, instance_id in sorted(M, key=lambda x: next(v.cost_per_interval for v in resource_pool if v.type_id == x[0])):
                st, eft, et = compute_EFT(task_id, vm_type, instance_id)
                temp_usage = vm_usage.copy()
                temp_usage[(vm_type, instance_id)] = max(temp_usage[(vm_type, instance_id)], eft)
                temp_used_vms = used_vms | {(vm_type, instance_id)}
                cost = compute_cost(temp_usage, temp_used_vms)
                
                if cost <= budget + 1e-10 and cost <= best_cost:
                    best_cost = cost
                    best_vm = (vm_type, instance_id)
                    best_st = st
                    best_eft = eft
                    best_et = et

        if best_vm:
            vm_type, instance_id = best_vm
            ft = best_st + best_et
            AL.append((task_id, best_vm, round(best_st, 2), round(ft, 2)))
            vm_usage[best_vm] = max(vm_usage[best_vm], ft)
            used_vms.add(best_vm)
            tasks[task_id].vm = best_vm
        else:
            print(f"Warning: No feasible VM found for task {task_id} within budget")
            return M, [], []

    Z = sorted_task_ids
    return M, AL, Z

def validate_schedule(AL, edges, budget, resource_pool, T=3600):
    """
    Validate the schedule for overlaps, dependencies, and budget compliance.
    """
    if not AL:
        return False, "Empty schedule"
    
    vm_schedules = {}
    for task_id, (vm_type, instance_id), st, ft in AL:
        vm_id = (vm_type, instance_id)
        if vm_id not in vm_schedules:
            vm_schedules[vm_id] = []
        vm_schedules[vm_id].append((st, ft))

    for vm_id, schedule in vm_schedules.items():
        sorted_schedule = sorted(schedule)
        for i in range(1, len(sorted_schedule)):
            if sorted_schedule[i][0] < sorted_schedule[i-1][1] - 1e-10:
                return False, f"Overlap detected on VM {vm_id} between {sorted_schedule[i-1]} and {sorted_schedule[i]}"

    task_map = {(tid, 0): (vm, st, ft) for tid, (vm, _), st, ft in AL}
    for child_id, parents in edges.items():
        if child_id not in task_map:
            continue
        child_st = task_map[(child_id, 0)][1]
        for parent_id in parents:
            if parent_id in task_map:
                parent_ft = task_map[(parent_id, 0)][2]
                if parent_ft > child_st + 1e-10:
                    return False, f"Dependency violation: {parent_id} (FT: {parent_ft}) must finish before {child_id} (ST: {child_st})"

    vm_usage = {}
    used_vms = set(vm_id for _, vm_id, _, _ in AL)
    for vm_id in used_vms:
        vm_type = vm_id[0]
        vm_usage[vm_id] = max(ft for _, v, _, ft in AL if v == vm_id)
    for vm in resource_pool:
        if vm.lease_finish_time > 0:
            vm_id = (vm.type_id, 0)
            vm_usage[vm_id] = max(vm_usage.get(vm_id, 0), vm.lease_finish_time)
    
    total_cost = sum(math.ceil(max(max_ft, 1e-10) / T) * next(v.cost_per_interval for v in resource_pool if v.type_id == vm_type)
                     for (vm_type, _), max_ft in vm_usage.items())
    if total_cost > budget + 1e-10:
        return False, f"Budget exceeded: Cost {total_cost} > Budget {budget}"

    return True, "Schedule is valid"

def plot_gantt_chart(AL, resource_pool):
    """
    Plot a Gantt chart of the task schedule.
    """
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
    vm_ids = sorted(set(vm_id for _, vm_id, _, _ in AL))
    for i, vm_id in enumerate(vm_ids):
        for task_id, (vm_type, instance_id), st, ft in AL:
            if (vm_type, instance_id) == vm_id:
                plt.barh(f"VM {vm_type}.{instance_id}", ft - st, left=st,
                         color=colors[i % len(colors)], edgecolor='black')
                plt.text(st, f"VM {vm_type}.{instance_id}", f"T{task_id}", va='center')
    plt.xlabel("Time (seconds)")
    plt.ylabel("VMs")
    plt.title("Gantt Chart of Task Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def grp_heft(tasks, edges, resource_pool, budget, bandwidth=20, T=3600):
    """
    GRP-HEFT algorithm (Sections 4.1.1 and 4.1.2).
    """
    M = greedy_resource_provisioning(resource_pool, budget, T)
    if not M:
        print("Warning: No VMs leased within budget")
        return
    
    M, AL, Z = modified_heft(tasks, edges, M, resource_pool, budget, bandwidth, T)
    
    if AL:
        makespan = max(ft for _, _, _, ft in AL)
        print(f"Makespan: {makespan}")
        is_valid, message = validate_schedule(AL, edges, budget, resource_pool, T)
        print(f"Validation: {message}")
        if is_valid:
            plot_gantt_chart(AL, resource_pool)
        
        X = {task_id: vm_id for task_id, vm_id, _, _ in AL}
        Y = list(set(vm_id[0] for vm_id in M if vm_id in [v for _, v, _, _ in AL] or next((vm.lease_finish_time for vm in resource_pool if vm.type_id == vm_id[0]), 0) > 0))
        print(f"Placement Vector X (Task-to-VM): {X}")
        print(f"Placement Vector Y (VM Types): {Y}")
        print(f"Placement Vector Z (Task Order): {Z}")
        
        with open("schedule_output.txt", 'w') as f:
            f.write("Task ID\tVM Type\tInstance ID\tStart Time\tFinish Time\n")
            for task_id, (vm_type, instance_id), st, ft in sorted(AL, key=lambda x: x[3]):
                f.write(f"{task_id}\t{vm_type}\t{instance_id}\t{st}\t{ft}\n")
        print("Schedule written to schedule_output.txt")
    else:
        print("Warning: No valid solution found.")

if __name__ == "__main__":
    dag_file_path = "/home/swayam-gupta/Downloads/ISEA/CYBERSHAKE/CYBERSHAKE.n.50.0.dag"
    tasks, edges = parse_dag_file(dag_file_path)
    budget = 9
    bandwidth = 20
    T = 3600
    resource_pool = [
        VM("m1.small", processing_capacity=1, cost_per_interval=0.044, bandwidth=100),
        VM("m1.medium", processing_capacity=2, cost_per_interval=0.087, bandwidth=200),
        VM("m3.medium", processing_capacity=3, cost_per_interval=0.067, bandwidth=200),
        VM("m1.large", processing_capacity=4, cost_per_interval=0.175, bandwidth=400),
        VM("m3.large", processing_capacity=6.5, cost_per_interval=0.133, bandwidth=400),
        VM("m1.xlarge", processing_capacity=8, cost_per_interval=0.350, bandwidth=800),
        VM("m3.xlarge", processing_capacity=13, cost_per_interval=0.266, bandwidth=800),
        VM("m3.2xlarge", processing_capacity=26, cost_per_interval=0.532, bandwidth=1000)
    ]
    grp_heft(tasks, edges, resource_pool, budget, bandwidth, T)

    # Test case mimicking Figures 4 and 5
    # tasks = {
    #     't_en': Task('t_en', 0),
    #     't1': Task('t1', 750),
    #     't2': Task('t2', 750),
    #     't3': Task('t3', 750),
    #     't_ex': Task('t_ex', 0)
    # }
    
    # edges = {
    #     't1': {'t_en': 0},
    #     't2': {'t1': 3.4},
    #     't3': {'t2': 3.4},
    #     't_ex': {'t3': 0}
    # }
    # tasks['t1'].predecessors = ['t_en']
    # tasks['t_en'].successors = ['t1']
    # tasks['t2'].predecessors = ['t1']
    # tasks['t1'].successors = ['t2']
    # tasks['t3'].predecessors = ['t2']
    # tasks['t2'].successors = ['t3']
    # tasks['t_ex'].predecessors = ['t3']
    # tasks['t3'].successors = ['t_ex']
    
    # budget = 0.3
    # bandwidth = 20
    # T = 3600
    # resource_pool = [
    #     VM("m1.medium", processing_capacity=3.75, cost_per_interval=0.2, bandwidth=200),
    #     VM("m1.small", processing_capacity=1.7, cost_per_interval=0.1, bandwidth=100)
    # ]
    # resource_pool[0].lease_finish_time = 3500
    
    # grp_heft(tasks, edges, resource_pool, budget, bandwidth, T)