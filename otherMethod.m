function otherMethod  %%other method
    warning('off');
    seed = 1;
    cross_validation(seed)    
end

function cross_validation(seed)
    CV=5;
    rand('state',seed);
    load extracted_interaction.txt;
    load extracted_lncRNA_expression.txt
    feature_matrix = extracted_lncRNA_expression';
    interaction_matrix = extracted_interaction;
    
    train_interaction_l = get_train_set(seed, interaction_matrix);
    train_interaction_p = get_train_set(seed, interaction_matrix');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    result = zeros(4,7);    %7 evaluation metrics
    for k = 1 : CV
        fprintf('begin to implement the cross validation:round =%d/%d\n', k, CV);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
        predict_matrix_rwr = RWR_matrix(train_interaction_l(:,:,k), 0.1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        predict_matrix_cf = CF_matrix(train_interaction_p(:,:,k));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        predict_matrix_ra = resource_allocate(train_interaction_l(:,:,k));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        predict_matrix_hrwr = heterogeneous_matrix(feature_matrix, train_interaction_p(:,:,k), 0.9, 0.9, 0.9);
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        result(1,:) = result(1,:) + model_evaluate(interaction_matrix,predict_matrix_rwr,train_interaction_l(:,:,k));
        result(2,:) = result(2,:) + model_evaluate(interaction_matrix',predict_matrix_cf,train_interaction_p(:,:,k));
        result(3,:) = result(3,:) + model_evaluate(interaction_matrix,predict_matrix_ra,train_interaction_l(:,:,k));
        result(4,:) = result(4,:) + model_evaluate(interaction_matrix', predict_matrix_hrwr, train_interaction_p(:,:,k));
        result/k                                                                  
    end
    result = result / CV
end

function result = get_train_set(seed, interaction_matrix)
    CV=5;
    [row,col]=size(interaction_matrix);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [row_index,col_index]=find(interaction_matrix==1);
    link_num=sum(sum(interaction_matrix)); 
    rand('state',seed);
    random_index=randperm(link_num);
    size_of_CV=round(link_num/CV);                                                      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    result = zeros(row, col, 5);
    for k=1:CV
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (k~=CV)
           test_row_index=row_index(random_index((size_of_CV*(k-1)+1):(size_of_CV*k)));
           test_col_index=col_index(random_index((size_of_CV*(k-1)+1):(size_of_CV*k)));
        else
          test_row_index=row_index(random_index((size_of_CV*(k-1)+1):end));
          test_col_index=col_index(random_index((size_of_CV*(k-1)+1):end));
        end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        train_set=interaction_matrix;
        test_link_num=size(test_row_index,1);
        for i=1:test_link_num
              train_set(test_row_index(i),test_col_index(i))=0;                 
        end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        result(:,:,k) = train_set;
    end
end

function result = model_evaluate(interaction_matrix,predict_matrix,train_ddi_matrix)
    real_score = interaction_matrix(:);
    predict_score = predict_matrix(:);
    index = train_ddi_matrix(:);
    test_index = find(index==0);
    real_score = real_score(test_index);
    predict_score = predict_score(test_index);
    aupr = AUPR(real_score,predict_score);
    auc = AUC(real_score,predict_score);
    [sen,spec,precision,accuracy,f1] = evaluation_metric(real_score,predict_score);
    result = [aupr,auc,sen,spec,precision,accuracy,f1];
end

function [sen,spec,precision,accuracy,f1] = evaluation_metric(interaction_score,predict_score)
    max_value = max(predict_score);
    min_value = min(predict_score);
    threshold = min_value+(max_value-min_value)*(1:999)/1000;
    for i = 1 : 999
       predict_label = (predict_score>threshold(i));
       [temp_sen(i),temp_spec(i),temp_precision(i),temp_accuracy(i),temp_f1(i)] = classification_metric(interaction_score,predict_label);
    end
    [max_score,index] = max(temp_f1);
    sen = temp_sen(index);
    spec = temp_spec(index);
    precision = temp_precision(index);
    accuracy = temp_accuracy(index);
    f1 = temp_f1(index);
end

function [sen,spec,precision,accuracy,f1] = classification_metric(real_label,predict_label)
    tp_index=find(real_label==1 & predict_label==1);
    tp=size(tp_index,1);

    tn_index=find(real_label==0 & predict_label==0);
    tn=size(tn_index,1);

    fp_index=find(real_label==0 & predict_label==1);
    fp=size(fp_index,1);

    fn_index=find(real_label==1 & predict_label==0);
    fn=size(fn_index,1);

    accuracy=(tn+tp)/(tn+tp+fn+fp);
    sen=tp/(tp+fn);
    recall=sen;
    spec=tn/(tn+fp);
    precision=tp/(tp+fp);
    f1=2*recall*precision/(recall+precision);
end

function area = AUPR(real,predict)
    max_value = max(predict);
    min_value = min(predict);

    threshold = min_value+(max_value-min_value)*(1:999)/1000;

    threshold = threshold';
    threshold_num = length(threshold);
    tn = zeros(threshold_num,1);
    tp = zeros(threshold_num,1);
    fn = zeros(threshold_num,1);
    fp = zeros(threshold_num,1);

    for i=1:threshold_num
        tp_index=logical(predict>=threshold(i) & real==1);
        tp(i,1)=sum(tp_index);

        tn_index=logical(predict<threshold(i) & real==0);
        tn(i,1)=sum(tn_index);

        fp_index=logical(predict>=threshold(i) & real==0);
        fp(i,1)=sum(fp_index);

        fn_index=logical(predict<threshold(i) & real==1);
        fn(i,1)=sum(fn_index);
    end

    sen=tp./(tp+fn);
    precision=tp./(tp+fp);
    recall=sen;
    x=recall;
    y=precision;
    [x,index]=sort(x);
    y=y(index,:);

    area=0;
    x(1,1)=0;
    y(1,1)=1;
    x(threshold_num+1,1)=1;
    y(threshold_num+1,1)=0;
    area=0.5*x(1)*(1+y(1));
    for i=1:threshold_num
        area=area+(y(i)+y(i+1))*(x(i+1)-x(i))/2;
    end
    % plot(x,y)
end

function area=AUC(real,predict)
    max_value=max(predict);
    min_value=min(predict);
    threshold=min_value+(max_value-min_value)*(1:999)/1000;
    threshold=threshold';
    threshold_num=length(threshold);
    tn=zeros(threshold_num,1);
    tp=zeros(threshold_num,1);
    fn=zeros(threshold_num,1);
    fp=zeros(threshold_num,1);
    for i=1:threshold_num
        tp_index=logical(predict>=threshold(i) & real==1);
        tp(i,1)=sum(tp_index);

        tn_index=logical(predict<threshold(i) & real==0);
        tn(i,1)=sum(tn_index);

        fp_index=logical(predict>=threshold(i) & real==0);
        fp(i,1)=sum(fp_index);

        fn_index=logical(predict<threshold(i) & real==1);
        fn(i,1)=sum(fn_index);
    end

    sen=tp./(tp+fn);
    spe=tn./(tn+fp);
    y=sen;
    x=1-spe;
    [x,index]=sort(x);
    y=y(index,:);
    [y,index]=sort(y);
    x=x(index,:);

    area=0;
    x(threshold_num+1,1)=1;
    y(threshold_num+1,1)=1;
    area=0.5*x(1)*y(1);
    for i=1:threshold_num
        area=area+(y(i)+y(i+1))*(x(i+1)-x(i))/2;
    end
    % plot(x,y)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function score_matrix=resource_allocate(interaction_matrix)
    [drug_num,sideffect_num]=size(interaction_matrix);
    resource_allocate_matrix=zeros(sideffect_num,sideffect_num);
    drug_degree=sum(interaction_matrix,2);
    sideffect_degree=sum(interaction_matrix,1);
    for i=1:sideffect_num
        for j=1:sideffect_num
            z=0;
           set1=find(interaction_matrix(:,i)==1);
           set2=find(interaction_matrix(:,j)==1);
           set = intersect(set1,set2);  
           if  ~isempty(set)
              num=size(set,1);
              for p=1:num
                if drug_degree(p)~=0
                  z=z+interaction_matrix(set(p,1),i)*interaction_matrix(set(p,1),j)/drug_degree(set(p,1));
                end
              end
           end

            if sideffect_degree(j)~=0
                resource_allocate_matrix(i,j)=z/sideffect_degree(j);
            end
        end
    end

    score_matrix=(resource_allocate_matrix*interaction_matrix')';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function score_matrix = RWR_matrix(interaction_matrix, c)
    similarity_matrix = get_similarity_matrix(interaction_matrix);
    %get transformation matrix, i.e. normalize the similarity matrix
    num = size(similarity_matrix, 2);
    for i = 1 : num
        similarity_matrix(i, i) = 0;
    end
    column_sum_matrix = sum(similarity_matrix);
    sum_diagonal_matrix = pinv(diag(column_sum_matrix));
    transformation_matrix =  similarity_matrix * sum_diagonal_matrix;
    %get initial state, same propability for each
    row_sum_interaction_matrix = pinv(sum(interaction_matrix, 2));
    initial_state_matrix = diag(row_sum_interaction_matrix) * interaction_matrix;
    score_matrix = pinv(eye(num) - c * transformation_matrix) * initial_state_matrix;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function score_matrix = CF_matrix(interaction_matrix)
    protein_similarity_matrix = get_similarity_matrix(interaction_matrix);
    %row-mormalize the protein_similarity_matrix
    row_sum_matrix = sum(protein_similarity_matrix, 2);
    sum_diagonal_matrix = pinv(diag(row_sum_matrix));
    row_normalized_protein_similarity_matrix = sum_diagonal_matrix * protein_similarity_matrix;
    score_matrix = row_normalized_protein_similarity_matrix * interaction_matrix;
end

function similarity_matrix = get_similarity_matrix(interaction_matrix)
    %get intersection matrix
    intersection_matrix = interaction_matrix * interaction_matrix';
    %get denominator matrix
    protein_degree_matrix = sum(interaction_matrix, 2);
    denominator_matrix = sqrt(protein_degree_matrix * protein_degree_matrix');
    %calculate similarity_matrix of protein
    similarity_matrix = intersection_matrix ./ denominator_matrix;
    similarity_matrix(isnan(similarity_matrix)) = 0;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rna_similarity_matrix = get_ran_similarity_matrix(interaction_matrix)
    rna_similarity_matrix = corrcoef(interaction_matrix);
    rna_similarity_matrix(isnan(rna_similarity_matrix)) = 0;
    rna_similarity_matrix = abs(rna_similarity_matrix);
end

function score_matrix = heterogeneous_matrix(feature_matrix, interaction_matrix, gamma, beta, delta)
    protein_similarity_matrix = get_protein_similarity_matrix(interaction_matrix);
    rna_similarity_matrix = get_ran_similarity_matrix(interaction_matrix);
    row_sum_i = sum(interaction_matrix, 2);
    col_sum_i = sum(interaction_matrix);
    %get protein transformation matrix
    protein_num = size(protein_similarity_matrix);
    for i = 1 : protein_num
        protein_similarity_matrix(i, i) = 0;
    end
    row_sum_matrix_p = sum(protein_similarity_matrix, 2);
    sum_diagonal_matrix_p = pinv(diag(row_sum_matrix_p));
    transformation_matrix_p =  sum_diagonal_matrix_p * protein_similarity_matrix .* (1 - gamma);
    transformation_matrix_p(row_sum_i == 0, :) = transformation_matrix_p(row_sum_i == 0, :) ./ (1 - gamma);
    %get lncRNA transformation matrix
    rna_num = size(rna_similarity_matrix);
    for i = 1 : rna_num
        rna_similarity_matrix(i, i) = 0;
    end
    row_sum_matrix_l = sum(rna_similarity_matrix, 2);
    sum_diagonal_matrix_l = pinv(diag(row_sum_matrix_l));
    transformation_matrix_l = sum_diagonal_matrix_l * rna_similarity_matrix .* (1 - gamma);
    transformation_matrix_l(:, col_sum_i == 0) = transformation_matrix_l(:, col_sum_i == 0) ./ (1 - gamma);
    %get protein-lncRNA transformation matrix and lncRNA-protein transformation matrix
    row_sum_matrix_pl = row_sum_i;
    sum_diagonal_matrix_pl = pinv(diag(row_sum_matrix_pl));
    transformation_matrix_pl = sum_diagonal_matrix_pl * interaction_matrix .* gamma;
    %%%
    row_sum_matrix_lp = col_sum_i;
    sum_diagonal_matrix_lp = pinv(diag(row_sum_matrix_lp));
    transformation_matrix_lp = sum_diagonal_matrix_lp * interaction_matrix' .* gamma;
    transformation_matrix_lp(isnan(transformation_matrix_lp)) = 0;
    transformation_matrix_pl(isnan(transformation_matrix_pl)) = 0;
    %get transformation matrix
    transformation_matrix = [transformation_matrix_p transformation_matrix_pl; transformation_matrix_lp transformation_matrix_l];
    %get initial state
    rna_initial_state = eye(rna_num) * (1 - beta);
    protein_initial_state = interaction_matrix *  sum_diagonal_matrix_lp * beta;
    initial_state_matrix = [protein_initial_state; rna_initial_state];
    %get score matrix
    num = size(transformation_matrix, 1);
    score_matrix = (eye(num) / ((eye(num) - (1 - delta) * transformation_matrix'))) * delta * initial_state_matrix;
    score_matrix = score_matrix([1 : protein_num], :);
    %test
%     transformation_matrix
%     initial_state_matrix
%     score_matrix
end

function protein_similarity_matrix = get_protein_similarity_matrix(interaction_matrix)
    %get intersection matrix
    intersection_matrix = interaction_matrix * interaction_matrix';
    %get denominator matrix
    protein_degree_matrix = sum(interaction_matrix, 2);
    denominator_matrix = sqrt(protein_degree_matrix * protein_degree_matrix');
    %calculate similarity_matrix of protein
    protein_similarity_matrix = intersection_matrix ./ denominator_matrix;
    protein_similarity_matrix(isnan(protein_similarity_matrix)) = 0;
end
