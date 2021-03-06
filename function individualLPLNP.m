function individualLPLNP  %%individual
    warning('off');
    seed = 1;
    load extracted_interaction.txt;
    %protein based LPLNP
    interaction_matrix = extracted_interaction';
    %interaction
    cross_validation(seed, interaction_matrix, 6, 0.5, [], 0);
    %protein ctd
    load protein_ctd;
    cross_validation(seed, interaction_matrix, 23, 0.3, protein_ctd, 1);
    
    %lncRNA based LPLNP
    interaction_matrix = extracted_interaction;
    %interaction
    cross_validation(seed, interaction_matrix, 100, 0.7, [], 0);
    %lncRNA sequence
    load extracted_lncRNA_sequence_CT.txt
    cross_validation(seed, interaction_matrix, 800, 0.1, extracted_lncRNA_sequence_CT, 1);
    %lncRNA expression
    load extracted_lncRNA_expression.txt
    cross_validation(seed, interaction_matrix, 100, 0.9, extracted_lncRNA_expression, 1);
end

%%if we use interaction profile as feature, set feature_matrix = [], flag = 0; otherwise, flag = 1
function cross_validation(seed, interaction_matrix, neighbor_num, alpha, feature_matrix, flag)
    CV=5; %%We take 5-fold crossvalidation
    rand('state',seed);
    [row,col]=size(interaction_matrix);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [row_index,col_index]=find(interaction_matrix==1); %%find all the elments that its value is 1
    link_num=sum(sum(interaction_matrix));  %% caculate the numbers of the interaction
    rand('state',seed);
    random_index=randperm(link_num);
    size_of_CV=round(link_num/CV);                                                      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    result=zeros(1,7);
    for k=1:CV
        fprintf('begin to implement the cross validation:round =%d/%d\n', k, CV);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (k~=CV) %% We allocate all the interaction elements into 5 parts
           test_row_index=row_index(random_index((size_of_CV*(k-1)+1):(size_of_CV*k)));
           test_col_index=col_index(random_index((size_of_CV*(k-1)+1):(size_of_CV*k)));
        else
          test_row_index=row_index(random_index((size_of_CV*(k-1)+1):end));
          test_col_index=col_index(random_index((size_of_CV*(k-1)+1):end));
        end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        train_set=interaction_matrix;
        test_link_num=size(test_row_index,1);
        for i=1:test_link_num %% let interaction elements' value equal to 0 in the test matrix
              train_set(test_row_index(i),test_col_index(i))=0;                 
        end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (flag == 0)
           	feature_matrix = train_set;
        end
        similairty_matrix=Label_Propagation(feature_matrix,0,neighbor_num,'regulation2');    
        predict_matrix_LP=calculate_labels(similairty_matrix,train_set,alpha);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        result=result+model_evaluate(interaction_matrix,predict_matrix_LP,train_set);
        result/k                                                                        
    end
    result=result/CV;
    result
end


function result=model_evaluate(interaction_matrix,predict_matrix,train_ddi_matrix) %% evaulate our prediction 
    real_score=interaction_matrix(:);
    predict_score=predict_matrix(:);
    index=train_ddi_matrix(:);
    test_index=find(index==0);
    real_score=real_score(test_index);
    predict_score=predict_score(test_index);
    aupr=AUPR(real_score,predict_score);
    auc=AUC(real_score,predict_score);
    [sen,spec,precision,accuracy,f1]=evaluation_metric(real_score,predict_score);
    result=[aupr,auc,sen,spec,precision,accuracy,f1];
end


function [sen,spec,precision,accuracy,f1]=evaluation_metric(interaction_score,predict_score)%%evaluate our prediction 
    max_value=max(predict_score);
    min_value=min(predict_score);
    threshold=min_value+(max_value-min_value)*(1:999)/1000;
    for i=1:999
       predict_label=(predict_score>threshold(i));
       [temp_sen(i),temp_spec(i),temp_precision(i),temp_accuracy(i),temp_f1(i)]=classification_metric(interaction_score,predict_label);
    end
    [max_score,index]=max(temp_f1);
    sen=temp_sen(index);
    spec=temp_spec(index);
    precision=temp_precision(index);
    accuracy=temp_accuracy(index);
    f1=temp_f1(index);
end


function [sen,spec,precision,accuracy,f1]=classification_metric(real_label,predict_label) %%label the elements
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

function area=AUPR(real,predict) %%calculate the value of AUPR
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

function area=AUC(real,predict) %%calculate the value of AUC
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



%%'regulation1':LN similarity, 'regulation2': RLN similarity
function W=optimization_similairty_matrix(feature_matrix,nearst_neighbor_matrix,tag,regulation) %%quadratic programming
   row_num=size(feature_matrix,1);
   W=zeros(1,row_num);
   if tag==1
       row_num=1;
   end
   for i=1:row_num
       nearst_neighbors=feature_matrix(logical(nearst_neighbor_matrix(i,:)'),:);   
       neighbors_num=size(nearst_neighbors,1);
       G1=repmat(feature_matrix(i,:),neighbors_num,1)-nearst_neighbors;
       G2=repmat(feature_matrix(i,:),neighbors_num,1)'-nearst_neighbors';
       if regulation=='regulation2'
         G_i=G1*G2+eye(neighbors_num);
       end
       if regulation=='regulation1'
         G_i=G1*G2;
       end
       H=2*G_i;
       f=[];
       A=[];
       if isempty(H)
           A;
       end
       
       b=[];
       Aeq=ones(neighbors_num,1)';
       beq=1;
       lb=zeros(neighbors_num,1);
       ub=[];
       options=optimset('Display','off');
       [w,fval]= quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);
       w=w';
       W(i,logical(nearst_neighbor_matrix(i,:)))=w;     
   end
end

function distance_matrix=calculate_instances(feature_matrix) %%calculate the distance between each feature vector of lncRNAs or proteins.
    [row_num,col_num]=size(feature_matrix);
    distance_matrix=zeros(row_num,row_num);
    for i=1:row_num
        for j=i+1:row_num
            distance_matrix(i,j)=sqrt(sum((feature_matrix(i,:)-feature_matrix(j,:)).^2));
            distance_matrix(j,i)=distance_matrix(i,j);
        end
        distance_matrix(i,i)=col_num;
    end
end

function nearst_neighbor_matrix=calculate_neighbors(distance_matrix,neighbor_num)%% calculate the nearest K neighbors
  [sv si]=sort(distance_matrix,2,'ascend');
  [row_num,col_num]=size(distance_matrix);
  nearst_neighbor_matrix=zeros(row_num,col_num);
  index=si(:,1:neighbor_num);
  for i=1:row_num
       nearst_neighbor_matrix(i,index(i,:))=1;
  end
end

function W=Label_Propagation(feature_matrix,tag,neighbor_num,regulation) %% Using the method of label propagation to predict the interaction
    distance_matrix=calculate_instances(feature_matrix);
    nearst_neighbor_matrix=calculate_neighbors(distance_matrix,neighbor_num);
    W=optimization_similairty_matrix(feature_matrix,nearst_neighbor_matrix,tag,regulation);
end

function F=calculate_labels(W,Y,alpha)
    F=(1-alpha)*pinv(eye(size(W,1))-alpha*W)*Y;
end


