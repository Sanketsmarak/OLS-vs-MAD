%% 2D synthetic section
rc=zeros(101,20); % 20 traces with number of samples in each trace is 101
m=40;n=20;        % Makes a wedge at a depth of 40 samples
rc(m,:)=ones(1,n)*1;
synthSeis=zeros(101,n);
Wavelet=s_create_wavelet({'type','Ricker'},{'frequencies',30},{'step',2},{'wlength',160});
Wavelet=Wavelet.traces;
for i=0:n-1
    rc(m+i,i+1)=1; % creates the wedge
    synthSeis(:,i+1)=conv(rc(:,i+1),Wavelet,'same'); % generates synthetic seismic through convolution of wavelet and reflection coefficients
end
nSample=size(synthSeis,1);nMin=0;nMax=5;
%% Basis Pursuit Least Squares Solution
alpha=1000;
[MD_RC, MD_Seis] = MatrixDictionary(nSample,nMin,nMax,Wavelet);
[m,n]=size(MD_Seis);
Gmatrix=[MD_Seis' alpha*eye(n)]; %Included Damping Factor
bvector=ones(n,1);
Est_RC=zeros(size(rc));
Est_Seis=zeros(size(rc));
for i=1:size(synthSeis,2)
    cvector=[-synthSeis(:,i); zeros(n,1)];
    %Define lower and upper bound
    lb=[zeros(m,1);zeros(n,1)];
    ub=[inf(m,1);ones(n,1)];
    [primal,obj,exitflag,output,dual]=linprog(cvector,[],[],Gmatrix,bvector,lb,ub);
    %Result
    x=dual.eqlin;
    Est_RC(:,i) = MD_RC*x;
    Est_Seis(:,i) = MD_Seis*x;
end

%% Basis Pursuit Least Absolute Solution
lambda=1e-5;

[MD_RC1, MD_Seis1] = MatrixDictionary(nSample,nMin,nMax,Wavelet);

[q1,e1]=size(MD_Seis1);

G1=MD_Seis1;

A1=[G1,-eye(q1),zeros(q1,e1);-G1,-eye(q1),zeros(q1,e1);eye(e1),zeros(e1,q1),-eye(e1);-eye(e1),zeros(e1,q1),-eye(e1)];
Est_RC1=zeros(size(rc));
Est_Seis1=zeros(size(rc));
for i=1:size(synthSeis,2)
    
    d1=synthSeis(:,i);

    B1=[d1;-d1;zeros(e1,1);zeros(e1,1)];

    f1=[zeros(e1,1);ones(q1,1);lambda*ones(e1,1)];

    [x1]=linprog(f1,A1,B1);

    x1=x1(1:e1);

    Est_RC1(:,i)=MD_RC1*x1;

    Est_Seis1(:,i)=MD_Seis1*x1;
end

%% Plot

subplot(321)
imagesc(rc);
title('Synthetic Wedge');
subplot(322)
imagesc(synthSeis);
title('Synthetic Seismic');
subplot(323);
imagesc(Est_RC);
title('Inverted Wedge for OLS');
subplot(324)
imagesc(Est_Seis);
title('Inverted Seismic for OLS');
subplot(325)
imagesc(Est_RC1);
title('Inverted Wedge for MAD');
subplot(326)
imagesc(Est_Seis1);
title('Inverted Seismic for MAD');
