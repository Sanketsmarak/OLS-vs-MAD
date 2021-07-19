%% Single synthetic trace
Wavelet=s_create_wavelet({'type','Ricker'},{'frequencies',30},{'step',2},{'wlength',80});
Wavelet=Wavelet.traces;
wlog=read_las_file('WDU-5_jgw.las');
wlog.curve_info{11,1}='aImp';
refl=s_log2reflectivity(wlog,2);
synthSeis=conv(refl.traces,Wavelet,'same');
nMin=0;nMax=5;lambda=5e-4;
synthSeis=synthSeis/norm(synthSeis);
Wavelet=Wavelet/norm(Wavelet);
nSample=size(synthSeis,1);

%% Basis Pursuit Least Squares Solution
alpha=100;
[MD_RC, MD_Seis] = MatrixDictionary(nSample,nMin,nMax,Wavelet);
[m,n]=size(MD_Seis);
Gmatrix=[MD_Seis' alpha*eye(n)]; %Included Damping Factor
bvector=ones(n,1);
cvector=[-synthSeis; zeros(n,1)];
%Define lower and upper bound
lb=[zeros(m,1);zeros(n,1)];
ub=[inf(m,1);ones(n,1)];
[primal,obj,exitflag,output,dual]=linprog(cvector,[],[],Gmatrix,bvector,lb,ub);
%Result
x=dual.eqlin;
r = MD_RC*x;
rw = MD_Seis*x;

corr1=corrcoef(refl.traces,r);

%% Basis Pursuit Least Absolute Solution

[MD_RC1, MD_Seis1] = MatrixDictionary(nSample,nMin,nMax,Wavelet);

[q1,e1]=size(MD_Seis1);

G1=MD_Seis1;

A1=[G1,-eye(q1),zeros(q1,e1);-G1,-eye(q1),zeros(q1,e1);eye(e1),zeros(e1,q1),-eye(e1);-eye(e1),zeros(e1,q1),-eye(e1)];

d1=synthSeis;

B1=[d1;-d1;zeros(e1,1);zeros(e1,1)];

f1=[zeros(e1,1);ones(q1,1);lambda(1)*ones(e1,1)];

[x1]=linprog(f1,A1,B1);

x1=x1(1:e1);

Est_RC1=MD_RC1*x1;

Est_Seis1=MD_Seis1*x1;

corr2=corrcoef(Est_RC1,refl.traces);

subplot(311)
stem(Est_RC1)
subplot(312)
stem(r)
subplot(313)
plot(synthSeis)
hold on 
plot(Est_Seis1);
hold on 
plot(rw);
legend('Original Seismic','Inverted Seismic from MAD','Inverted Seismic from OLS');
