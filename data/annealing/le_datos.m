% annealing
printf('lendo problema annealing ...\n');

n_entradas= 38; n_clases= 5; n_fich= 2; fich{1}= 'anneal.data'; n_patrons(1)= 798; fich{2}= 'anneal.test'; n_patrons(2)= 100;

n_max= max(n_patrons); x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

continua=[0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0];

family = {'--','GB','GK','GS','TN','ZA','ZF','ZH','ZM','ZS'}; n_family=length(family);
product_type={ 'C', 'H', 'G'}; n_product_type=length(product_type);
steel={'-','R','A','U','K','M','S','W','V'};n_steel= length(steel);
temper_rolling={'-','T'};n_temper_rolling= length(temper_rolling);
condition={'-','S','A','X'};n_condition= length(condition);
formability={'-','1','2','3','4','5'};n_formability= length(formability);
non_ageing=	{'-','N'};n_non_ageing = length(non_ageing);
surface_finish={'P','M','-'};n_surface_finish = length(surface_finish);
surface_quality={'-','D','E','F','G'};n_surface_quality = length(surface_quality);
enamelability={'-','1','2','3','4','5'}; n_enamelability = length(enamelability);
bc={'Y','-'};n_bc = length(bc);
bf={'Y','-'}; n_bf = length(bf);
bt={'Y','-'};n_bt = length(bt);
bw_me={'B','M','-'}; n_bw_me = length(bw_me);
bl={'Y','-'};n_bl = length(bl);
m={'Y','-'};n_m = length(m);
chrom={'C','-'}; n_chrom = length(chrom);
phos={'P','-'}; n_phos = length(phos);
cbond={'Y','-'}; n_cbond = length(cbond);
marvi={'Y','-'}; n_marvi = length(marvi);
exptl={'Y','-'}; n_exptl = length(exptl);
ferro={'Y','-'};n_ferro = length(ferro);
corr={'Y','-'};n_corr = length(corr);
color={'B','R','V','C','-'}; n_color = length(color);
lustre={'Y','-'}; n_lustre = length(lustre);
jurofm={'Y','-'}; n_jurofm = length(jurofm);
s={'Y','-'};n_s = length(s);
p={'Y','-'}; n_p = length(p);
shape={'COIL', 'SHEET'}; n_shape = length(shape);
oil={'-','Y','N'}; n_oil = length(oil);
bore={'0000','0500','0600','0760'}; n_bore = length(bore);
packing={'-','1','2','3'}; n_packing = length(packing);
classes={'1','2','3','5','U'};  % non hai patróns da clase '4'

n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  t = fscanf(f,'%s',1);
%  	  printf('%s ', t);	  
	  if strcmp(t, '?')  % entrada ausente
		x(i_fich, i, j) = 0;
	  elseif 1==continua(j)
		x(i_fich, i, j) = str2double(t);
	  else
		if j==1
		  n= n_family; k=family;
		elseif j==2
		  n= n_product_type; k=product_type;
		elseif j==3
		  n= n_steel; k=steel;
		elseif j==6
		  n = n_temper_rolling; k= temper_rolling;
		elseif j==7  
		  n= n_condition; k=condition;
  		elseif j==8
		  n= n_formability; k=formability;
		elseif j==10
		  n= n_non_ageing; k= non_ageing;
		elseif j==11
			  n= n_surface_finish; k= surface_finish;
		elseif j==12
			  n= n_surface_quality; k= surface_quality;
		elseif j==13
			  n= n_enamelability; k= enamelability;
		elseif j==14
			  n= n_bc; k= bc;
		elseif j==15
			  n= n_bf; k= bf;
		elseif j==16
			  n= n_bt; k= bt;
		elseif j==17
			  n= n_bw_me; k= bw_me;
		elseif j==18,
			  n= n_bl; k= bl;
		elseif j==19
			  n= n_m; k= m;
		elseif j==20
			  n= n_chrom; k= chrom;
		elseif j==21
			  n= n_phos; k= phos;
		elseif j==22
			  n= n_cbond; k= cbond;
		elseif j==23
			  n= n_marvi; k= marvi;
		elseif j==24
			  n= n_exptl; k= exptl;
		elseif j==25
			  n= n_ferro; k= ferro;
		elseif j==26
			  n= n_corr; k= corr;
		elseif j==27
			  n= n_color; k= color;
		elseif j==28
			  n= n_lustre; k= lustre;
		elseif j==29
			  n= n_jurofm; k= jurofm;
		elseif j==30
			  n= n_s; k= s;
		elseif j==31
			  n= n_p; p={'Y','-'};
		elseif j==32
			  n= n_shape; k= shape;
		elseif j==36
			  n= n_oil; k= oil;
		elseif j==37
			  n= n_bore; k= bore;
		elseif j==38
			  n= n_packing; k= packing;
		else
		  error('entrada %i non discreta', j);
		end
		a = 2/(n-1); b= (1+n)/(1-n);
		for l=1:n
		  if strcmp(t, k(l))
			x(i_fich,i,j) = a*l + b; break
		  end
		end
	  end
	end
	% lectura da clase
	t= fscanf(f,'%s',1);
	for j=1:n_clases
	  if strcmp(t, classes(j))
		cl(i_fich,i)= j-1; break
	  end
	end
%  	printf('cl= %i\n', cl(i_fich,i));
%  	disp(x(i_fich,i,:))	
  end
  fclose(f);
end
