import numpy as np
import torch
import math
import copy
import re
device = "cpu"
train_device= torch.device('cpu')
def get_representation(program_json, schedule_json):
    max_dims= 8
    max_accesses = 15 # TODO: check if 10 is enough
    program_representation = []
    indices_dict = dict()
    computations_dict = program_json['computations']
    ordered_comp_list = sorted(list(computations_dict.keys()), key = lambda x: computations_dict[x]['absolute_order'])
    skew = 0
    interchange = 0
    skew_params=[0]*4
    interchange_params=[0,0]
    iteration_domain=[]
    class LargeAccessMatices(Exception):
        pass
    i=0
    # Interchange representation
    interchanged = 0
    skewed = 0
    
    for index, comp_name in enumerate(ordered_comp_list):
        #print("comp_name:")
        i_skew = 0
        #print(comp_name)
        comp_dict = computations_dict[comp_name]
        comp_schedule_dict = schedule_json[comp_name]
        depth = len(comp_dict['iterators'])
        if len(comp_dict['accesses'])>max_accesses:
#             print('too much acc')
            raise LargeAccessMatices
        if len(comp_dict['accesses'])<1:
#             print('too little acc')
            raise LargeAccessMatices
        comp_representation = []
        # Is this computation a reduction 
        comp_representation.append(+comp_dict['comp_is_reduction'])


#         iterators representation + tiling and interchage
        iterators_repr = []
        for iterator_name in comp_dict['iterators']:
            
            iterator_dict = program_json['iterators'][iterator_name]
            iterators_repr.append(iterator_dict['upper_bound']) 
#            iterators_repr.append(iterator_dict['lower_bound'])
            # unfuse schedule replacing the lower bound for compability issue, this enables the use transfer learning from an older model 
            parent_iterator = program_json['iterators'][iterator_name]['parent_iterator']
            """
            if iterator_name in schedule_json[comp_name]['interchange_dims']:
                iterators_repr.append(1) #interchanged true
            else:
                iterators_repr.append(0) #interchanged false
            
            # Skewing representation
            skewed = 0
            skew_factor = 0
            skew_extent = 0
            if schedule_json[comp_name]['skewing'] and (iterator_name in schedule_json[comp_name]['skewing']['skewed_dims']):
                skewed = 1 #skewed: true
                skew_factor_index = schedule_json[comp_name]['skewing']['skewed_dims'].index(iterator_name)
                skew_factor = int(schedule_json[comp_name]['skewing']['skewing_factors'][skew_factor_index]) # skew factor
                skew_extent = int(schedule_json[comp_name]['skewing']['average_skewed_extents'][skew_factor_index]) # skew extent
            iterators_repr.append(skewed)
            iterators_repr.append(skew_factor)
            iterators_repr.append(skew_extent)
            """
             # Parallelization representation
            parallelized = 0
            if iterator_name == schedule_json[comp_name]['parallelized_dim']:
                parallelized = 1 # parallelized true
            iterators_repr.append(parallelized)
            
            if (schedule_json[comp_name]['tiling']!={}):
                if iterator_name in schedule_json[comp_name]['tiling']['tiling_dims']:
                    iterators_repr.append(1) #tiled: true
                    tile_factor_index = schedule_json[comp_name]['tiling']['tiling_dims'].index(iterator_name)
                    iterators_repr.append(int(schedule_json[comp_name]['tiling']['tiling_factors'][tile_factor_index])) #tile factor
                else:
                    iterators_repr.append(0) #tiled: false
                    iterators_repr.append(0) #tile factor 0
            else: #tiling = None
                iterators_repr.append(0) #tiled: false
                iterators_repr.append(0) #tile factor 0 
            key="transformation_matrix"
            if key not in comp_schedule_dict.keys():
                if iterator_name in comp_schedule_dict['interchange_dims']:
                    interchanged = 1 # interchanged = True
                    interchange_params[i]=int(iterator_name[-1])
    
                     
                    i+=1
                #comp_repr[placeholders_indices_dict[l_code+'Interchanged']] = interchanged
                if comp_schedule_dict['skewing'] and (iterator_name in comp_schedule_dict['skewing']['skewed_dims']):
                    skew=1
                    skewed = 1 #skewed: true
                    skew_factor = -1
                    skew_extent = -1
                    
                    skew_factor_index = comp_schedule_dict['skewing']['skewed_dims'].index(iterator_name)
                    skew_factor = int(comp_schedule_dict['skewing']['skewing_factors'][skew_factor_index]) # skew factor

                    skew_params[i_skew]=int(iterator_name[-1])
                    skew_extent = int(comp_schedule_dict['skewing']['average_skewed_extents'][skew_factor_index]) # skew extent
                    skew_params[i_skew+2]=skew_factor
#                    
                    i_skew+=1
            # is this dimension saved (this dimension does not disapear aftre reduction)
#             iterators_repr.append(+(iterator_name in comp_dict['real_dimensions']))
        
            
        iterator_repr_size = int(len(iterators_repr)/len(comp_dict['iterators']))
        iterators_repr.extend([0]*iterator_repr_size*(max_dims-len(comp_dict['iterators']))) # adding iterators padding 
        
        
        key="fusions"
        if key in schedule_json.keys():
            if schedule_json['fusions']!= None:
                iterators_repr.append(1) #fused true
            else:
                iterators_repr.append(0) #fused false
        else:
            iterators_repr.append(0)
        
        max_depth = 5
        
        padded_matrix = np.zeros((max_depth, max_depth))
        

        inter_matrix = np.identity(len(comp_dict['iterators'])).tolist()
        if(interchanged):
            #print("interchange_params")
            #print(interchange_params[0])
            #print(interchange_params[1])
            inter_matrix = get_schedule_matrix_interchange(interchange_params[0],interchange_params[1],len(comp_dict['iterators']))
            #print(inter_matrix)

        #skew_matrix = [[0] * len(comp_dict['iterators'])] * len(comp_dict['iterators'])
        #for i in range(len(comp_dict['iterators'])):
        #    skew_matrix[i][i]=1
        skew_matrix = np.identity(len(comp_dict['iterators'])).tolist()

        if(skewed):
            
            skew_matrix = get_schedule_matrix_skewing(skew_params[0], skew_params[1], skew_params[2], skew_params[3],   len(comp_dict['iterators']))
            
                
        
        key="transformation_matrix"
        if key in comp_schedule_dict.keys():
            if(len(comp_schedule_dict['transformation_matrix'])>1):
                for i in range(max_depth):
                    for j in range(max_depth):
                        if (i<depth and j<depth ):
                            padded_matrix[i][j] = int(comp_schedule_dict['transformation_matrix'][i*depth+j])
        else:
            transformation_matrix = np.matmul(np.array(skew_matrix), np.array(inter_matrix) ).tolist()
            # Adding padding so that the trandormations for each loop always start at the same index
            for i in range(max_depth):
                for j in range(max_depth):
                    if (i<depth and j<depth):
                        padded_matrix[i][j] = transformation_matrix[i][j]
                     
        #print(schedule_json[comp_name]['transformation_matrix'])
        
              
        
        #padded_matrix[max_depth-1][max_depth-1]=55
        padded_matrix = np.array(padded_matrix).flatten().tolist()
        iterators_repr.extend(padded_matrix)
        comp_representation.extend(iterators_repr) #adding the iterators representation    
       
        #       write access represation 
        write_access_matrix = isl_to_write_matrix(comp_dict['write_access_relation'])
        write_access_matrix = np.array(write_access_matrix)
        write_access_matrix = np.c_[np.ones(write_access_matrix.shape[0]), write_access_matrix] # adding tags for marking the used rows
        write_access_matrix = np.r_[[np.ones(write_access_matrix.shape[1])], write_access_matrix] # adding tags for marking the used columns
        padded_write_matrix = np.zeros((max_dims + 1, max_dims + 2))
        padded_write_matrix[:write_access_matrix.shape[0],:write_access_matrix.shape[1]-1] = write_access_matrix[:,:-1] #adding padding to the access matrix
        padded_write_matrix[:write_access_matrix.shape[0],-1] = write_access_matrix[:,-1] #adding padding to the access matrix
        write_access_repr = [comp_dict['write_buffer_id']+1] + padded_write_matrix.flatten().tolist()
        comp_representation.extend(write_access_repr)
        
#         accesses representation
        accesses_repr=[]
        for access_dict in comp_dict['accesses']:
            access_matrix = access_dict['access_matrix']
            access_matrix = np.array(access_matrix)
            padded_access_matrix = np.zeros((max_dims, max_dims + 1))
            padded_access_matrix[:access_matrix.shape[0],:access_matrix.shape[1]-1] = access_matrix[:,:-1] #adding padding to the access matrix
            padded_access_matrix[:access_matrix.shape[0],-1] = access_matrix[:,-1] #adding padding to the access matrix
            access_repr = [access_dict['buffer_id']] + padded_access_matrix.flatten().tolist() # input_id + flattened access matrix 
            # is this access a reduction (the computation is accesing itself)
            access_repr.append(+access_dict['access_is_reduction'])
            accesses_repr.extend(access_repr)

        #access_repr_len = max_dims*(max_dims + 1)
        access_repr_len = max_dims*(max_dims + 1) + 1 +1 #+1 for input id, +1 for is_access_reduction
        accesses_repr.extend([0]*access_repr_len*(max_accesses-len(comp_dict['accesses']))) #adding accesses padding
    
        comp_representation.extend(accesses_repr) #adding access representation

#         operation histogram
        comp_representation.append(comp_dict['number_of_additions'])
        comp_representation.append(comp_dict['number_of_subtraction'])
        comp_representation.append(comp_dict['number_of_multiplication'])
        comp_representation.append(comp_dict['number_of_division'])

        
#         unrolling representation
        if (schedule_json[comp_name]['unrolling_factor']!=None):
            comp_representation.append(1) #unrolled True
            comp_representation.append(int(schedule_json[comp_name]['unrolling_factor'])) #unroll factor
        else:
            comp_representation.append(0) #unrolled false
            comp_representation.append(0) #unroll factor 0

        # adding log(x+1) of the representation
#         log_rep = list(np.log1p(comp_representation))
#         comp_representation.extend(log_rep)
        
        program_representation.append(comp_representation)
        #print("program_representation : ")
        #print(program_representation)
        indices_dict[comp_name] = index
    
    # transforming the schedule_json inorder to have loops as key instead of computations, this dict helps building the loop vectors
    loop_schedules_dict = dict()
    for loop_name in program_json['iterators']:
        loop_schedules_dict[loop_name]=dict()
        """
        loop_schedules_dict[loop_name]['interchanged']=False
        loop_schedules_dict[loop_name]['interchanged_with']=None
        loop_schedules_dict[loop_name]['skewed']=False
        loop_schedules_dict[loop_name]['skewed_dims']=None
        loop_schedules_dict[loop_name]['skew_factor']=None
        loop_schedules_dict[loop_name]['skew_extent']=None
        """
        loop_schedules_dict[loop_name]['parallelized']=False
        loop_schedules_dict[loop_name]['tiled']=False
        loop_schedules_dict[loop_name]['tile_depth']=None
        loop_schedules_dict[loop_name]['tiled_dims']=None
        loop_schedules_dict[loop_name]['tile_factor']=None
        loop_schedules_dict[loop_name]['unrolled']=False
        loop_schedules_dict[loop_name]['unroll_factor']=None
        loop_schedules_dict[loop_name]['unroll_comp']=None
        loop_schedules_dict[loop_name]['fused']=False 
        
    #loop_schedules_dict['transformation_matrix']=np.zeros((max_depth, max_depth))    
    for comp_name in schedule_json:
        if not comp_name.startswith('comp'): 
            continue # skip the non computation keys
        """
        if schedule_json[comp_name]['transformation_matrix']:
            max_depth = 5 
            depth = len(comp_dict['iterators'])
            padded_matrix = np.zeros((max_depth, max_depth))
            if(len(schedule_json[comp_name]['transformation_matrix'])>1):
                for i in range(max_depth):
                    for j in range(max_depth):
                        if (i<depth and j<depth ):
                            padded_matrix[i][j] = int(schedule_json[comp_name]['transformation_matrix'][(i*depth)+j])
            padded_matrix = np.array(padded_matrix).flatten().tolist()
            loop_schedules_dict['transformation_matrix']=padded_matrix
        """                                                                                                                        
        
        
    

        if schedule_json[comp_name]['parallelized_dim']:
            loop_schedules_dict[schedule_json[comp_name]['parallelized_dim']]['parallelized']=True
        if schedule_json[comp_name]['tiling']!={}:
            for tiled_loop_index,tiled_loop in enumerate(schedule_json[comp_name]['tiling']['tiling_dims']):
                loop_schedules_dict[tiled_loop]['tiled']=True
                loop_schedules_dict[tiled_loop]['tile_depth']=schedule_json[comp_name]['tiling']['tiling_depth']
                loop_schedules_dict[tiled_loop]['tiled_dims']=schedule_json[comp_name]['tiling']['tiling_dims']
                loop_schedules_dict[tiled_loop]['tile_factor']=int(schedule_json[comp_name]['tiling']['tiling_factors'][tiled_loop_index])
#         if schedule_json[comp_name]['unrolling_factor']!=None:
#             comp_innermost_loop=computations_dict[comp_name]['iterators'][-1] 
#             tiling_dims = [] if schedule_json[comp_name]['tiling']=={} else schedule_json[comp_name]['tiling']['tiling_dims']
            #interchange_dims =schedule_json[comp_name]['interchange_dims']
            
#             if (not ((comp_innermost_loop in tiling_dims)or(comp_innermost_loop in interchange_dims))):# unrolling always applied to innermost loop, if tilling or interchange is applied to innermost, unroll is applied to the resulting loop instead of the orginal, hence we don't represent it
#                 loop_schedules_dict[comp_innermost_loop]['unrolled']=True
#                 loop_schedules_dict[comp_innermost_loop]['unroll_factor']=int(schedule_json[comp_name]['unrolling_factor'])
#                 loop_schedules_dict[comp_innermost_loop]['unroll_comp']=comp_name
            
    
    # collect the set of iterators that are used for computation (to eleminate those that are only used for inputs)
    #print(loop_schedules_dict)
    real_loops = set()
    for comp_name in computations_dict:
        real_loops.update(computations_dict[comp_name]['iterators'])
        
    #building loop tensor
    loops_representation_list = []
    loops_indices_dict = dict()
    loop_index=0
    real_iterators=[]
    def get_real_iterators(node):   
        real_iterators.append(node['loop_name'])
        if node['child_list']!=[]:
            for child_node in node['child_list']:
                get_real_iterators(child_node)
    
    get_real_iterators(schedule_json['tree_structure'])
    key="fusions"
    if key in schedule_json.keys():
        if schedule_json['fusions']!= None:
            #print(schedule_json['fusions'][0])
            loop_schedules_dict[real_iterators[schedule_json['fusions'][0][2]]]['fused']=True
            
    for loop_name in real_iterators:
        if not (loop_name in real_loops): # this removes the iterators that are only used for decraling inputs
            continue
        loop_representation=[]
        loop_dict = program_json['iterators'][loop_name]
        # upper and lower bound
        loop_representation.append(loop_dict['upper_bound'])
        loop_representation.append(loop_dict['lower_bound'])

        """
        if loop_schedules_dict[loop_name]['interchanged']:
            loop_representation.append(1) #interchanged True
        else:
            loop_representation.append(0) #interchanged False     
        if loop_schedules_dict[loop_name]['skewed']:
            loop_representation.append(1) #skewed True
            loop_representation.append(loop_schedules_dict[loop_name]['skew_factor']) #skew factor
            loop_representation.append(loop_schedules_dict[loop_name]['skew_extent']) #skew extent
        else:
            loop_representation.append(0) # skewed false
            loop_representation.append(0) # factor
            loop_representation.append(0) # extent
        """
        if loop_schedules_dict[loop_name]['parallelized']:
            loop_representation.append(1) #parallelized True
        else:
            loop_representation.append(0) # parallelized false
        if loop_schedules_dict[loop_name]['tiled']:
            loop_representation.append(1) #tiled True
            loop_representation.append(loop_schedules_dict[loop_name]['tile_factor']) #tile factor
        else:
            loop_representation.append(0) #tiled False
            loop_representation.append(0) #tile factor 0
        # TODO: check if unroll representation should be moved to comp vector instead of loop vector
#         if loop_schedules_dict[loop_name]['unrolled']:
#             loop_representation.append(1) #unrolled True
#             loop_representation.append(loop_schedules_dict[loop_name]['unroll_factor']) #unroll factor
#         else:
#             loop_representation.append(0) #unrolled False
#             loop_representation.append(0) #unroll factor 0
        # adding log(x+1) of the loop representation
        loop_log_rep = list(np.log1p(loop_representation))
        loop_representation.extend(loop_log_rep)
        loops_representation_list.append(loop_representation)    
        loops_indices_dict[loop_name]=loop_index
        loop_index+=1
 
    #loop_representation.extend(loop_schedules_dict['transformation_matrix'])
    #print("loop_representation")
    #print(loop_representation)
    def update_tree_atributes(node):     
        node['loop_index'] = torch.tensor(loops_indices_dict[node['loop_name'][:3]]).to(train_device)
        if node['computations_list']!=[]:
            node['computations_indices'] = torch.tensor([indices_dict[comp_name] for comp_name in node['computations_list']]).to(train_device)
            node['has_comps'] = True
        else:
            node['has_comps'] = False
        for child_node in node['child_list']:
            update_tree_atributes(child_node)
        return node
    
    tree_annotation = copy.deepcopy(schedule_json['tree_structure']) #to avoid altering the original tree from the json
    prog_tree = update_tree_atributes(tree_annotation) 
    #print("loops_representation_list")
    #print(loops_representation_list)
    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_representation_list),0)#.to(device)
    computations_tensor = torch.unsqueeze(torch.FloatTensor(program_representation),0)#.to(device)     
    #print(loops_tensor)
    return prog_tree, computations_tensor, loops_tensor
def isl_to_write_matrix(isl_map): # for now this function only support reductions
    comp_iterators_str = re.findall(r'\[(.*)\]\s*->', isl_map)[0]
    buffer_iterators_str = re.findall(r'->\s*\w*\[(.*)\]', isl_map)[0]
    buffer_iterators_str=re.sub(r"\w+'\s=","",buffer_iterators_str)
    comp_iter_names = re.findall(r'(?:\s*(\w+))+', comp_iterators_str)
    buf_iter_names = re.findall(r'(?:\s*(\w+))+', buffer_iterators_str)
    matrix = np.zeros([len(buf_iter_names),len(comp_iter_names)+1])
    for i,buf_iter in enumerate(buf_iter_names):
        for j,comp_iter in enumerate(comp_iter_names):
            if buf_iter==comp_iter:
                matrix[i,j]=1
                break
    return matrix
def isl_to_write_dims(isl_map): # return the buffer iterator that defines the write buffer
    buffer_iterators_str = re.findall(r'->\s*\w*\[(.*)\]', isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=","",buffer_iterators_str)
    buf_iter_names = re.findall(r'(?:\s*(\w+))+', buffer_iterators_str)
    return buf_iter_names

def get_schedule_matrix_interchange(first_loop, second_loop, dim):
    matrix = np.identity(dim).tolist()
    matrix[first_loop][second_loop] = 1
    matrix[second_loop][first_loop] = 1
    matrix[second_loop][second_loop] = 0
    matrix[first_loop][first_loop] = 0
    return matrix

def get_schedule_matrix_skewing(first_loop, second_loop, first_skew, second_skew, dim):
    matrix = np.identity(dim).tolist()
    matrix[first_loop][second_loop] = second_skew
    matrix[first_loop][first_loop] = first_skew
    return matrix