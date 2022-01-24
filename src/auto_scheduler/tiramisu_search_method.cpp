#include <tiramisu/auto_scheduler/search_method.h>
#include <random>
#include <functional>
#include <fstream>
#include <iostream>

namespace tiramisu::auto_scheduler
{
 std::string get_name_ast_expr_isl( isl_ast_expr *expr);
void beam_search::search(syntax_tree& ast)
{
    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
        ast.clear_new_optimizations();
       
    std::vector<syntax_tree*> children;
        
    // Look for an optimization that can be applied
    int nb_optims_tried = 0;
    int nb_explored_optims = ast.nb_explored_optims;
    
    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
    {
        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
        
        children = scheds_gen->generate_schedules(ast, optim_type);
        
        nb_explored_optims++;
        nb_optims_tried++;
    }
      
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;
       
    // Evaluate children and sort them from smallest to highest evaluation
    

   // evaluate while removing illegal versions
    auto iterator = children.begin();
    while (iterator != children.end())
    {
        (*iterator)->nb_explored_optims = nb_explored_optims;
        (*iterator)->transform_ast();
        
        if ((*iterator)->ast_is_legal() == false) {
            // print deleted Ast 
            (*iterator)->print_previous_optims();
            std::cout << "\n-----------" << std::endl;
            (*iterator)->print_new_optims();
            (*iterator)->print_ast();
            (*iterator)->print_isl_states();
            std::cout << "\n<illegal>\n";
            delete (*iterator);
            iterator = children.erase(iterator);
        }
        else {
            // evaluate and print Ast 
            (*iterator)->evaluation = eval_func->evaluate(*(*iterator));

            (*iterator)->print_previous_optims();
            std::cout << "\n-----------" << std::endl;
            (*iterator)->print_new_optims();
            (*iterator)->print_ast();
            std::cout << "Evaluation : " << (*iterator)->evaluation << std::endl << std::endl;
            (*iterator)->print_isl_states();
            (*iterator)->print_computations_accesses();
            std::cout << "\n<legal>\n";

            if ((*iterator)->evaluation < best_evaluation)
            {
                best_evaluation = (*iterator)->evaluation;
                best_ast = (*iterator);
            }

            ++iterator;

        }
        
        nb_explored_schedules++;
    }

    // Stop if we reached the maximum depth
    if (nb_explored_optims >= max_depth)
        return ;
        
    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    ast_copy->nb_explored_optims = nb_explored_optims;
    children.push_back(ast_copy);

    // Sort children from smallest evaluation to largest

    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });

    // keep the top 'beam_size' children and delete the rest
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];

    children.resize(std::min(beam_size, (int)children.size()));
    
    // Search recursively on the best children
    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;        
        search(*child);
    }
}

void beam_search::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    std::default_random_engine rand_generator;

    //if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
    //    ast.clear_new_optimizations();

    std::vector<syntax_tree*> children;

    // Look for an optimization that can be applied
    int nb_optims_tried = 0;
    int nb_explored_optims = ast.nb_explored_optims;

    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
    {
        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
        children = scheds_gen->generate_schedules(ast, optim_type);

        nb_explored_optims++;
        nb_optims_tried++;
    }
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;

    // Evaluate children and sort them from smallest to highest evaluation
    // evaluate while removing illegal versions
    auto iterator = children.begin();
    while (iterator != children.end())
    {
        syntax_tree *child = *iterator;
        child->nb_explored_optims = nb_explored_optims;
        child->transform_ast();

        if (child->schedule_is_prunable()){
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                // print deleted Ast
                child->print_previous_optims();
                std::cout << "\n-----------" << std::endl;
                child->print_new_optims();
                child->print_ast();
                std::cout << "\n<Schedule pruned>\n";
            }
            delete child;
            iterator = children.erase(iterator);
        }

        else if (!child->ast_is_legal()) {
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                // print deleted Ast
                child->print_previous_optims();
                std::cout << "\n-----------" << std::endl;
                child->print_new_optims();
                child->print_ast();
                child->print_isl_states();
                std::cout << "\n<illegal>\n";
            }
            delete child;
            iterator = children.erase(iterator);
        }
        else {

            // print and evaluate Ast

            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                child->print_previous_optims();
                std::cout << "\n-----------" << std::endl;
                child->print_new_optims();
                child->print_ast();
                child->print_isl_states();
                std::cout << "\n<legal>\n";
                child->print_computations_accesses();
            }

            std::vector<float> measurements;
            if (child->can_set_default_evaluation()) { // if yes the child's evaluation is set to a default value
                measurements = {child->evaluation};
            }
            else{
                measurements = exec_eval->get_measurements_matrix(*child, false, schedule_timeout);
                child->evaluation = min_eval(measurements);
            }

            parent_trace->add_child_path(child, schedules_annotations->size());

            std::string schedule_annot = evaluate_by_learning_model::get_schedule_json(*child);

            //remove the last two characters }\n
            schedule_annot.pop_back();
            schedule_annot.pop_back();

            if (std::isfinite(child->evaluation)) // the evaluation is not finite mean that the schedule didn't run
                schedule_annot += ", \n\"execution_times\" : " + measurements_to_str(measurements) + "\n}\n";
            else
                schedule_annot += ", \n\"execution_times\" : null\n}\n";

            schedules_annotations->push_back(schedule_annot);
                
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                std::ofstream myfile;
                myfile.open ("example.txt",std::ios_base::app);
                myfile<<"Schedule: "<<schedule_annot<<std::endl;
                myfile<<"Schedule number "<< schedules_annotations->size() << std::endl;
                myfile<<"Evaluation : " << child->evaluation << std::endl;
                myfile<<"Schedule number "<< schedules_annotations->size() << std::endl;
                myfile.close();
                std::cout << "Schedule number "<< schedules_annotations->size() << std::endl;
                std::cout << "Evaluation : " << child->evaluation << std::endl;
                std::cout << "Number of measurements : " << measurements.size() << std::endl;
                std::cout << "===================================" << std::endl << std::endl;
            }

            if (std::isinf(child->evaluation))
                std::cerr<< "Evaluation of schedule "<< schedules_annotations->size() <<" failed "<< std::endl;

            if (child->evaluation < best_evaluation)
            {
                best_evaluation = child->evaluation;
                best_ast = child;
            }

            ++iterator;

        }

        nb_explored_schedules++;
    }

    // Stop if we reached the maximum depth
    if (nb_explored_optims >= max_depth)
        return ;

    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    ast_copy->nb_explored_optims = nb_explored_optims;
    children.push_back(ast_copy);
    parent_trace->add_child_path(ast_copy, parent_trace->get_candidate_id()); // keeps the same id since it's just copy

    // Sort children from smallest evaluation to largest
//    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
    // shuffle the children so that they are selected a random
    std::shuffle(std::begin(children), std::end(children), rand_generator);

    // keep the top 'beam_size' children and delete the rest
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];

    children.resize(std::min(beam_size, (int)children.size()));

    // Search recursively on the best children
    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;
        search_save(*child, schedules_annotations, parent_trace->child_mappings[child], schedule_timeout);
    }
}
int determinant( std::vector <  std::vector<int> >  matrix, int n) {
   int det = 0;
   int o;
   if(n==1) return matrix.at(0).at(0);
   std::vector <  std::vector<int> >  submatrix(n);
   for(o = 0; o<n; o++){
                submatrix.at(o)= std::vector<int> (n);}
   if (n == 2)
   return ((matrix.at(0).at(0) * matrix.at(1).at(1)) - (matrix.at(1).at(0) * matrix.at(0).at(1)));
   else {
      for (int x = 0; x < n; x++) {
         int subi = 0;
         for (int i = 1; i < n; i++) {
            int subj = 0;
            for (int j = 0; j < n; j++) {
               if (j == x)
               continue;
               submatrix.at(subi).at(subj) = matrix.at(i).at(j);
               subj++;
            }
            subi++;
         }
         det = det + (pow(-1, x) * matrix.at(0).at(x) * determinant( submatrix, n - 1 ));
      }
   }
   return det;
}
bool check_if_repeated( std::vector < std::vector<int> >  matrix,std::vector < std::vector < std::vector<int> > > matrices)
{
    //if there are no matrices to compare to then we return false
    if(matrices.size()==0) return false;
    
    int depth = matrix.size();
    int i=0;
    while(i<matrices.size()){ 
        //for each matrix that we already explored  
        bool diffrent = false;
        for (int s=0;s<depth;s++){
                for (int d=0;d<depth;d++){
                            // set diffrent to true if we find one element that is not the same
                            if (matrix.at(s).at(d)!=matrices.at(i).at(s).at(d)) diffrent =true ;
                    }
        }
        //if we found the same matrix return true
        if (!diffrent) return true;
    i++;
    }

     
    return false;
}
std::vector < std::vector < std::vector<int> > > beam_search::get_random_matrcies(int nb_out_matrcies, int depth)
{
    std::vector <std::vector <  std::vector<int> >>  result(nb_out_matrcies);
    int nb_valid_matrices = 0;
    int max_depth = 6;
    if (depth>max_depth) std::cout << "WARNING: the depth of this program is too big. Matrix generation will take a long time \n"<< std::endl;
    srand((unsigned) time(0));
    while(nb_valid_matrices<nb_out_matrcies)
    {
        std::vector <  std::vector<int> >  random(depth);
        bool valid = false;
        while (!valid)
        {   
            //std::cout << "generating";
            int l, c;
            std::vector <  std::vector<int> >  randomL(depth);
            for(l = 0; l<depth; l++){
                randomL.at(l)= std::vector<int>(depth);
                for(c = 0; c<depth; c++){
                                if (l>c){
                                    randomL.at(l).at(c) = (rand() %10) - 5;
                                }else{
                                    randomL.at(l).at(c) = 0;
                                }
                }
            }
             
            std::vector <  std::vector<int> >  randomU(depth);
            for(l = 0; l<depth; l++){
                randomU.at(l)= std::vector<int>(depth);
                for(c = 0; c<depth; c++){
                            if (l<c){
                                randomU.at(l).at(c) = (rand() % 10) - 5;
                            }else{
                                randomU.at(l).at(c) = 0;
                            }
                }
            }
            for(l = 0; l<depth; l++){
                randomL.at(l).at(l) =1;
                randomU.at(l).at(l) =1;
                int sum=0,j=0;
                if(l<depth-1){
                    sum=0;
                    for(j=0;j<l;j++){
                        sum+= randomL.at(l).at(j) * randomU.at(j).at(depth-1);
                    }
                    randomU.at(l).at(depth-1) = (1-sum)/randomL.at(l).at(l);
                }else{
                    sum=0;
                    for(j=1;j<l+1;j++){
                        sum+= randomL.at(l).at(j) * randomU.at(j).at(depth-1);
                    }
                    randomL.at(depth-1).at(0) = (1-sum)/randomU.at(0).at(depth-1);
                }
            }
            int k;
            for(l = 0; l < depth; ++l){
                random.at(l)= std::vector<int>(depth);
                for(c = 0; c < depth; ++c)
                    {
                        for(k = 0; k < depth; ++k)
                        {
                            random.at(l).at(c)+= randomL.at(l).at(k) * randomU.at(k).at(c);
                        }
                    }
            }
            // Reduce elements into [-7, 7] values
            int mx = 7;
            int m = -1000;
            int x,y;
            for(int i = 0; i < depth; i++){
                        for(int j = 0; j< depth; j++){
                            if(m < std::abs(random.at(i).at(j)))
                            {
                                m = random.at(i).at(j);
                                x = i;
                                y = j;
                            }
                        }
            }
            int mm=-8;
            int useless = 0;
            int steps = 0;
            bool bad_case;
            int f = -1;
            // Check determinant equals 1
            //std::cout<<"Before \n"<< determinant(random, depth)<<std::endl;
            while(m>=mx && useless<= depth*depth){
                    steps+=1;
                    bad_case=0;
                    for(int i = 0; i < depth; i++){
                        if (i==x) continue;
                        if(random.at(i).at(y)!=0)
                            {
                                f = i;
                                break;
                            }
                        if(i==depth-1) bad_case=1;
                    }
                    //std::cout<<"First \n"<<std::endl;
                    if(bad_case) break;
                    int s =1;
                    //std::cout<<f<<" Second \n"<<y<<std::endl;
                    if (m*random.at(f).at(y)<0) s=s*-1;
                    
                    for(int j = 0; j < depth; j++){
                            
                            random.at(x).at(j) = random.at(x).at(j) - s * random.at(f).at(j);
                    }
                    //std::cout<<"Second \n"<<std::endl;
                    mm=-10000;
                    for(int i = 0; i < depth; i++){
                        for(int j = 0; j< depth; j++){
                            if(mm < std::abs(random.at(i).at(j)))
                            {
                                mm = random.at(i).at(j);
                                x = i;
                                y = j;
                            }
                        }
                    }
                    //std::cout<<"Third \n"<<std::endl;
                    if(m>=mm) useless++;
                    m=mm;
            }
            // Check determinant equals 1
            int det = determinant(random, depth);
            bool det_bool = det==1 && !bad_case && (useless <= depth*depth) ;
            //std::cout<< "After \n"<<det_bool<<std::endl;
            // Check upper right determinants equal 1
            bool all_1 = true;
            if (det_bool){
                //std::cout << depth;
                int d=0,s=0;
                
                for (k=depth-1;k>0;k--){
                        //std::cout<<" sub matrix at depth: " <<k <<"\n";
                        std::vector <  std::vector<int> >  submatrixd(depth-k);
                        
                        for (s=0;s<depth-k;s++){
                                    submatrixd.at(s) = std::vector<int> (depth-k);
                                    for (d=0;d<depth-k;d++){
                                            
                                            submatrixd.at(s).at(d) = random.at(s).at(k+d); 
                                    } 
                        }  
                        
                        if(determinant(submatrixd, depth-k)!=1){ 
                            all_1 = false;
                            /*std::cout<< "failed at size: "<< depth-k<<"with determinant being:"<< determinant(submatrixd, depth-k) <<"\n";
                            for (s=0;s<depth-k;s++){
                                for (d=0;d<depth-k;d++){
                                            std::cout<< submatrixd.at(s).at(d) <<"\n";
                                    }
                            }*/
                        }
                        if(!all_1) break;
                }
            } 
            valid = det_bool && all_1 ;
        }
    //std::cout<< "got one done \n"<<std::endl;
    /*std::cout<< "starts \n";
    for(int i = 0; i < depth; i++){
                        for(int j = 0; j< depth; j++){
                                std::cout<<random.at(i).at(j)<<"\n"<<std::endl;
                             
                        }
            }
    std::cout<< "end \n";*/
    result.at(nb_valid_matrices) = random;
    nb_valid_matrices++;
    }
    return result;
}
/*
Generate one random matrix that verifies the conditions of: 1- determinant is one 2- all of the upper left determinants are 1
*/
  std::vector < std::vector<int> >  beam_search::get_random_matix(int depth)
{
    int max_depth = 6;
    if (depth>max_depth) std::cout << "WARNING: the depth of this program is too big. Matrix generation will take a long time \n"<< std::endl;
    std::vector <  std::vector<int> >  random(depth);
    bool valid = false;
    while (!valid)
    {   
        //generate a lower traiangular matrix 
        int l, c;
        std::vector <  std::vector<int> >  randomL(depth);
        for(l = 0; l<depth; l++){
            randomL.at(l)= std::vector<int>(depth);
            for(c = 0; c<depth; c++){
                            if (l>c){
                                randomL.at(l).at(c) = (rand() %16) - 8;
                            }else{
                                randomL.at(l).at(c) = 0;
                            }
            }
        }
        ////generate an upper traiangular matrix
        std::vector <  std::vector<int> >  randomU(depth);
        for(l = 0; l<depth; l++){
            randomU.at(l)= std::vector<int>(depth);
            for(c = 0; c<depth; c++){
                        if (l<c){
                            randomU.at(l).at(c) = (rand() % 16) - 8;
                        }else{
                            randomU.at(l).at(c) = 0;
                        }
            }
        }
        
        for(l = 0; l<depth; l++){
            randomL.at(l).at(l) =1;
            randomU.at(l).at(l) =1;
            int sum=0,j=0;
            if(l<depth-1){
                sum=0;
                for(j=0;j<l;j++){
                    sum+= randomL.at(l).at(j) * randomU.at(j).at(depth-1);
                }
                randomU.at(l).at(depth-1) = (1-sum)/randomL.at(l).at(l);
            }else{
                sum=0;
                for(j=1;j<l+1;j++){
                    sum+= randomL.at(l).at(j) * randomU.at(j).at(depth-1);
                }
                randomL.at(depth-1).at(0) = (1-sum)/randomU.at(0).at(depth-1);
            }
        }
        int k;
        //multiply both matrices and put the result in a matrix random
        for(l = 0; l < depth; ++l){
            random.at(l)= std::vector<int>(depth);
            for(c = 0; c < depth; ++c)
                {
                    for(k = 0; k < depth; ++k)
                    {
                        random.at(l).at(c)+= randomL.at(l).at(k) * randomU.at(k).at(c);
                    }
                }
        }
        
        // Reduce elements into [-7, 7] values
        int mx = 10;
        int m = -1000;
        int x,y;
        for(int i = 0; i < depth; i++){
                    for(int j = 0; j< depth; j++){
                        if(m < std::abs(random.at(i).at(j)))
                        {
                            m = random.at(i).at(j);
                            x = i;
                            y = j;
                        }
                    }
        }
        int mm=-8;
        int useless = 0;
        int steps = 0;
        bool bad_case;
        int f = -1;
        while(m>=mx && useless<= depth*depth){
                steps+=1;
                bad_case=0;
                for(int i = 0; i < depth; i++){
                    if (i==x) continue;
                    if(random.at(i).at(y)!=0)
                        {
                            f = i;
                            break;
                        }
                    if(i==depth-1) bad_case=1;
                }
                
                if(bad_case) break;
                int s =1;
                
                if (m*random.at(f).at(y)<0) s=s*-1;
                
                for(int j = 0; j < depth; j++){
                        
                        random.at(x).at(j) = random.at(x).at(j) - s * random.at(f).at(j);
                }
                
                mm=-10000;
                for(int i = 0; i < depth; i++){
                    for(int j = 0; j< depth; j++){
                        if(mm < std::abs(random.at(i).at(j)))
                        {
                            mm = random.at(i).at(j);
                            x = i;
                            y = j;
                        }
                    }
                }

                if(m>=mm) useless++;
                m=mm;
        }
        // Check determinant equals 1
        int det = determinant(random, depth);
        // if determinant is 1 and we were able to reduce the elements to the desired interval
        bool det_bool = det==1 && !bad_case && (useless <= depth*depth) ;
        
        // Check upper right determinants equal 1
        bool all_1 = true;
        if (false){
            
            int d=0,s=0;
            
            for (k=depth-1;k>0;k--){
                    
                    std::vector <  std::vector<int> >  submatrixd(depth-k);
                    
                    for (s=0;s<depth-k;s++){
                                submatrixd.at(s) = std::vector<int> (depth-k);
                                for (d=0;d<depth-k;d++){
                                        
                                        submatrixd.at(s).at(d) = random.at(s).at(k+d); 
                                } 
                    }  
                    
                    if(determinant(submatrixd, depth-k)!=1){ 
                        all_1 = false;
                        break;
                    }
            }
        } 
        valid = det_bool && all_1 ;
    }
    return random;
    
}

static const char *op_str[] = {
	[isl_ast_op_and] = "and",
	[isl_ast_op_and_then] = "and_then",
	[isl_ast_op_or] = "or",
	[isl_ast_op_or_else] = "or_else",
	[isl_ast_op_max] = "max",
	[isl_ast_op_min] = "min",
	[isl_ast_op_minus] = "minus",
	[isl_ast_op_add] = "add",
	[isl_ast_op_sub] = "sub",
	[isl_ast_op_mul] = "mul",
	[isl_ast_op_div] = "div",
	[isl_ast_op_fdiv_q] = "fdiv_q",
	[isl_ast_op_pdiv_q] = "pdiv_q",
	[isl_ast_op_pdiv_r] = "pdiv_r",
	[isl_ast_op_zdiv_r] = "zdiv_r",
	[isl_ast_op_cond] = "cond",
	[isl_ast_op_select] = "select",
	[isl_ast_op_eq] = "eq",
	[isl_ast_op_le] = "le",
	[isl_ast_op_lt] = "lt",
	[isl_ast_op_ge] = "ge",
	[isl_ast_op_gt] = "gt",
	[isl_ast_op_call] = "call",
	[isl_ast_op_access] = "access",
	[isl_ast_op_member] = "member",
	[isl_ast_op_address_of] = "address_of"
};
/**
 * @brief Get the names of the iterator in the ast and save them into a map 
 * 
 * @param node 
 * @param isl_ast 
 * @param corr_map 
 */
void get_save_name_node(ast_node * node,std::vector<std::string> isl_ast,std::map <std::string,std::string>* corr_map, int &k){

    if(k>=isl_ast.size()){}
    else{
        (*corr_map).insert(std::pair<std::string,std::string> (isl_ast[k],node->name));
        k++;
    }
    for (ast_node *child : node->children)
    {
        get_save_name_node(child,isl_ast,corr_map,k);
    }
}
/**
 * @brief Get expression info from the ISL AST
 * 
 * @param expr 
 * @return std::string 
 */
    std::string get_name_arguments(isl_ast_expr *expr)
    {
        int i, n;
        std::string p;
        n = isl_ast_expr_get_op_n_arg(expr);

        if (n < 0) return "$Error in getting corr map arguments";
        if (n == 0) return "$Error in getting corr map arguments";

        for (i = 0; i < n; ++i) {
            isl_ast_expr *arg;
            arg = isl_ast_expr_get_op_arg(expr, i);
            if(i!=0) p = p + "," + get_name_ast_expr_isl(arg);
            else p = p + get_name_ast_expr_isl(arg);
            isl_ast_expr_free(arg);     
        }     
        return p;
    }
   std::string get_name_ast_expr_isl( isl_ast_expr *expr)
    {
        enum isl_ast_expr_type type;
        enum isl_ast_op_type op;
        isl_id *id;
        std::string p;

        if (!expr){return "!Expression";}  
        else{     
        type = isl_ast_expr_get_type(expr);
        switch (type) {
            case isl_ast_expr_error: return "$Error in the expression"; break;
                
            case isl_ast_expr_op:
                //std::cout<<"Entreing OP \n";
                op = isl_ast_expr_get_op_type(expr);
                if (op == isl_ast_op_error) return "$Opertation in getting the type of the operation";
                p = p + op_str[op] + "(";
                p = p + get_name_arguments(expr);
                p = p + ")";
                break;
            case isl_ast_expr_id:
                id = isl_ast_expr_get_id(expr);
                p = isl_id_get_name(id);
                break;
            case isl_ast_expr_int:
                //std::cout<<"Entreing Int \n";
                //v = isl_ast_expr_get_val(expr);
                break;
            default: return "%";
         }
        return p;
        }    
    }
/**
 * @brief Get the corr map from isl ast map
 * 
 * @param ast 
 * @return ** std::map <std::string,std::string>* 
 */
std::map <std::string,std::string>* get_corr_map_from_isl(syntax_tree& ast){
    //Create map between ISL iterator names and the AST iterator names
    isl_ast_expr * iter_expr;
    int stop=0;
    std::map <std::string,std::string>* corr_map= new std::map<std::string, std::string>();
    std::vector<std::string> isl_ast;  
    //Genrate isl ast
    ast.fct->gen_isl_ast();
        
    isl_ast_node *ast_i = ast.fct->get_isl_ast(); 
        
    //Get the iterator names from the ISL ast
    //Fill a vector with the iterator names 
    while(stop != 1)
    {  
        if(isl_ast_node_get_type(ast_i)==isl_ast_node_for){   
            iter_expr=isl_ast_node_for_get_iterator(ast_i);
            isl_ast.push_back(get_name_ast_expr_isl(iter_expr));          
            ast_i= isl_ast_node_for_get_body(ast_i); //n
        }else{
            stop = 1;
        }
    }  
    //Get the names of iterators of the AST and create the map corr_map
    int starting_k = 0;
    for (ast_node *root : ast.roots)
        {
            get_save_name_node(root,isl_ast,corr_map,starting_k);
        }
    return corr_map;
}
void beam_search::search_save_matrix(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    std::default_random_engine rand_generator;


    std::vector<syntax_tree*> children;
    std::vector<syntax_tree*> to_be_explored;

    // Look for an optimization that can be applied
    int nb_optims_tried = 0;
    int nb_explored_optims = ast.nb_explored_optims;
    
    //Generate n matrice asts to be explored
    //To change the number of matrices being explored go to: generate_schedules then the MATRIX case and change the length of the loop
    optimization_type optim_type = optimization_type::MATRIX;
    children = scheds_gen->generate_schedules(ast, optim_type);
    
    
    // Stop if no more optimizations can be applied
    //Add the current AST to the list of children
    if (children.size() == 0)
        return ;
    
    // Evaluate children and sort them from smallest to highest evaluation
    // evaluate while removing illegal versions
    auto iterator = children.begin();
    std::vector < std::vector < std::vector<int> > > matrices;

    std::map <std::string,std::string>* corr_map;

    // Add the corr_map to the ast structue
    corr_map = get_corr_map_from_isl(ast);
    //Hash the program string to get a unique seed for each program 
    std::hash<std::string> hasher;
    auto hashed = hasher(evaluate_by_learning_model::get_program_json(ast));
    srand(hashed);
    int nb_matrices =0;
    int nb_steps = 0;
    bool illegal = false;
    bool first_time_illegal = true;
    syntax_tree *child = *iterator;
    
    while (iterator != children.end())
    {
            
        //If we tried to find a new matrix too many times, we give up and explore the ones we found so far 
        if (nb_steps++>MAX_NB_STEPS){
            break;
        } 
        if (!illegal)  child = *iterator;

        // Add the corr_map to the ast structue
        child->corr_map = corr_map;
        child->nb_explored_optims = nb_explored_optims;
        
        int shape = child->get_program_depth();
        //save an AST in case the matrix is illegal
        syntax_tree* new_ast = new syntax_tree();
        new_ast = child->copy_ast();
        
        
        //add the matrix to optim.info
        child->new_optims.back().matrix = get_random_matix(shape);
        std::cout<<"la matrice #################\n";
        for (int i = 0; i < child->new_optims.back().matrix.size(); i++) {
        for (int j = 0; j < child->new_optims.back().matrix[i].size(); j++)
            std::cout << child->new_optims.back().matrix[i][j] << " ";
        std::cout << std::endl;
    }
        if(check_if_repeated(child->new_optims.back().matrix, matrices)) continue;
        
        matrices.push_back(child->new_optims.back().matrix);
        nb_matrices++;
        
        child->transform_ast();

        if (child->schedule_is_prunable()){
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                // print deleted Ast
                child->print_previous_optims();
                std::cout << "\n-----------" << std::endl;
                child->print_new_optims();
                child->print_ast();
                std::cout << "\n<Schedule pruned>\n";
            }
            delete child;
            iterator = children.erase(iterator);
        }
        else if (!child->program_is_legal()) {
            
            illegal=true;
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                // print deleted Ast
                child->print_previous_optims();
                std::cout << "\n-----------" << std::endl;
                child->print_new_optims();
                child->print_ast();
                child->print_isl_states();
                std::cout << "\n<illegal>\n";
            }
            
            if (first_time_illegal) {
                delete child;
                //iterator = children.erase(iterator);
                //if(iterator == children.end()) iterator--;
                first_time_illegal=false;
            }

            child = new_ast;
        }
        else {

            
            ++iterator;
            
            first_time_illegal=true;
            illegal = false;
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                child->print_previous_optims();
                std::cout << "\n-----------" << std::endl;
                child->print_new_optims();
                child->print_ast();
                child->print_isl_states();
                std::cout << "\n<legal>\n";
            }
            std::vector<float> measurements;
        
            if (child->can_set_default_evaluation()) { // if yes the child's evaluation is set to a default value
            
                measurements = {child->evaluation};
            }
            else{
                measurements = exec_eval->get_measurements_matrix(*child, false, schedule_timeout);
                child->evaluation = min_eval(measurements);
            }
            
            parent_trace->add_child_path(child, schedules_annotations->size());

            std::string schedule_annot = evaluate_by_learning_model::get_schedule_json(*child);

            //remove the last two characters }\n
            schedule_annot.pop_back();
            schedule_annot.pop_back();

            if (std::isfinite(child->evaluation)) // the evaluation is not finite mean that the schedule didn't run
                schedule_annot += ", \n\"execution_times\" : " + measurements_to_str(measurements) + "\n}\n";
            else
                schedule_annot += ", \n\"execution_times\" : null\n}\n";

            schedules_annotations->push_back(schedule_annot);

            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                std::ofstream myfile;
                myfile.open ("example.txt",std::ios_base::app);
                myfile<<"\n\n\n Schedule: "<<schedule_annot<<std::endl;
                myfile<<"Schedule number "<< schedules_annotations->size() << std::endl;
                myfile<<"Evaluation : " << child->evaluation << std::endl;
                myfile<<"Schedule number "<< schedules_annotations->size() << std::endl;
                myfile.close();
                std::cout << "Schedule number "<< schedules_annotations->size() << std::endl;
                std::cout << "Evaluation : " << child->evaluation << std::endl;
                std::cout << "Number of measurements : " << measurements.size() << std::endl;
                std::cout << "===================================" << std::endl << std::endl;
            }

            if (std::isinf(child->evaluation))
                std::cerr<< "Evaluation of schedule "<< schedules_annotations->size() <<" failed "<< std::endl;

            if (child->evaluation < best_evaluation)
            {
                best_evaluation = child->evaluation;
                best_ast = child;
            }
            to_be_explored.push_back(child);
        }
    }
    
    to_be_explored.resize(std::min(nb_matrices, (int)to_be_explored.size()));
    

    // Stop if we reached the maximum depth
    if (nb_explored_optims >= max_depth)
        return ;


    // Sort children from smallest evaluation to largest
    std::sort(to_be_explored.begin(), to_be_explored.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });

    // shuffle the children so that they are selected a random
    std::shuffle(std::begin(to_be_explored), std::end(to_be_explored), rand_generator);

    // keep the top 'beam_size' children and delete the rest
    for (int i = beam_size; i < to_be_explored.size(); ++i)
       delete to_be_explored[i];

    to_be_explored.resize(std::min(beam_size, (int)to_be_explored.size()));

    // Search recursively on the best children
    for (syntax_tree *child : to_be_explored)
    {
        child->search_depth = ast.search_depth ;
        
        search_save(*child, schedules_annotations, parent_trace->child_mappings[child], schedule_timeout);
    }
}

void mcts::search(syntax_tree& ast)
{
    std::default_random_engine rand_generator;
    
    std::vector<syntax_tree*> samples;
    std::vector<syntax_tree*> children;
    std::vector<double> children_evals;
    
    for (int epoch = 0; epoch < nb_samples; ++epoch)
    {
        // Starting from the initial ast, generate optimizations until reaching max_depth
        syntax_tree *ast_sample = &ast;
        for (int depth = 0; depth < max_depth; ++depth)
        {
            optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[depth % NB_OPTIMIZATIONS];
            children = scheds_gen->generate_schedules(*ast_sample, optim_type);
                        
            if (children.empty())
                continue;
                
            children_evals.clear();
            
            for (syntax_tree *child : children)
            {
                child->transform_ast();
                    
                child->evaluation = eval_func->evaluate(*child);
                children_evals.push_back(child->evaluation);
                
                nb_explored_schedules++;
            }
            
            // Add the current AST to the list of children
            children.push_back(ast_sample->copy_ast());
            children_evals.push_back(ast_sample->evaluation);
            
            // Sample an AST
            std::discrete_distribution<int> dist(children_evals.begin(), children_evals.end());
            ast_sample = children[dist(rand_generator)];
            
            samples.push_back(ast_sample);
        }
    }
    
    if (samples.empty())
        return ;
    
    // Sort schedules with respect to evaluations
    std::sort(samples.begin(), samples.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });
    
    // Execute top-k schedules and return the best
    for (int i = 0; i < topk; ++i)
    {
        float exec_time = exec_eval->evaluate(*samples[i]);
        if (exec_time < best_evaluation)
        {
            best_evaluation = exec_time;
            best_ast = samples[i];
        }
    }
}

void mcts::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    std::cerr<< "mcts::search_save not yet implemented" << std::endl;
    exit(1);
}
void mcts::search_save_matrix(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    std::cerr<< "mcts::search_save not yet implemented" << std::endl;
    exit(1);
}

// -------------------------------------------------------------------------- //

void beam_search_topk::search(syntax_tree& ast)
{
    // Do a beam search
    beam_search_subroutine(ast);
    
    // Sort schedules found
    std::sort(schedules.begin(), schedules.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });
    
    // Execute top-k schedules to find the best
    for (int i = 0; i < topk; ++i)
    {
        float exec_time = exec_eval->evaluate(*schedules[i]);
        if (exec_time < best_evaluation)
        {
            best_evaluation = exec_time;
            best_ast = schedules[i];
        }
    }
}

void beam_search_topk::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    std::cerr<< "beam_search_topk::search_save not yet implemented" << std::endl;
    exit(1);
}
void beam_search_topk::search_save_matrix(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    std::cerr<< "beam_search_topk::search_save not yet implemented" << std::endl;
    exit(1);
}

void beam_search_topk::beam_search_subroutine(syntax_tree& ast)
{
    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
        ast.clear_new_optimizations();
       
    std::vector<syntax_tree*> children;
        
    // Look for an optimization that can be applied
    int nb_optims_tried = 0;
    int nb_explored_optims = ast.nb_explored_optims;
    
    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
    {
        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
        children = scheds_gen->generate_schedules(ast, optim_type);
        
        nb_explored_optims++;
        nb_optims_tried++;
    }
       
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;
       
    // Evaluate children and sort them from smallest to highest evaluation
    for (syntax_tree *child : children)
    {
        child->nb_explored_optims = nb_explored_optims;
        child->transform_ast();
            
        child->evaluation = eval_func->evaluate(*child);
        
        nb_explored_schedules++;
    }
        
    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    ast_copy->nb_explored_optims = nb_explored_optims;
    children.push_back(ast_copy);

    // Sort children from smallest evaluation to largest
    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });
    
    for (int i = 0; i < beam_size; ++i)
        schedules.push_back(children[i]);
    
    // Stop if we reached the maximum depth
    if (nb_explored_optims >= max_depth)
        return ;

    // Search recursively on the best children
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
        
    children.resize(std::min(beam_size, (int)children.size()));

    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;        
        search(*child);
    }
}

void beam_search_accuracy_evaluator::search(syntax_tree& ast)
{
    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
        ast.clear_new_optimizations();
       
    std::vector<syntax_tree*> children;
        
    // Look for an optimization that can be applied
    int nb_optims_tried = 0;
    int nb_explored_optims = ast.nb_explored_optims;
    
    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
    {
        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
        children = scheds_gen->generate_schedules(ast, optim_type);
        
        nb_explored_optims++;
        nb_optims_tried++;
    }
       
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;
       
    // Evaluate children and sort them from smallest to highest evaluation
    for (syntax_tree *child : children)
    {
        child->nb_explored_optims = nb_explored_optims;
        child->transform_ast();
            
        child->evaluation = eval_func->evaluate(*child);
        
        // We evaluate both by the model and by execution
        model_evals_list.push_back(child->evaluation);
        exec_evals_list.push_back(exec_eval->evaluate(*child));
        
        if (child->evaluation < best_evaluation)
        {
            best_evaluation = child->evaluation;
            best_ast = child;
        }
        
        nb_explored_schedules++;
    }
    
    // Stop if we reached the maximum depth
    if (nb_explored_optims >= max_depth)
        return ;
        
    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    ast_copy->nb_explored_optims = nb_explored_optims;
    children.push_back(ast_copy);

    // Sort children from smallest evaluation to largest
    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });

    // Search recursively on the best children
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
        
    children.resize(std::min(beam_size, (int)children.size()));

    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;        
        search(*child);
    }
}

}
