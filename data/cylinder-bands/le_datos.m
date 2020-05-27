printf('lendo problema %s ...\n', problema);

n_entradas= 35; n_clases= 2; n_fich= 1; fich{1}= 'bands.data'; n_patrons(1)= 512;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
   	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j=1:4
	  fscanf(f,'%s',1); fscanf(f,'%c',1);   % timestamp, cylinder number, customer job number
	end
	for j = 1:n_entradas
	  t = fscanf(f,'%c',1);
	  if t == '?'
		x(i_fich,i,j) = 0; fscanf(f,'%c',1);
		continue;
	  end
	  fseek(f,-1,SEEK_CUR);
	  if j==1 || j==3 || j==8 || j==10
		val={'NO','YES'};
	  elseif j==2
		val={'KEY','TYPE'};
	  elseif j==4
		val={'BENTON', 'DAETWYLER', 'UDDEHOLM'};
	  elseif j==5
		val={'GALLATIN', 'WARSAW', 'MATTOON'};
	  elseif j==6
		val={'UNCOATED', 'COATED', 'SUPER'};
	  elseif j==7
		val={'UNCOATED', 'COATED', 'COVER'};
	  elseif j==9
		val={'XYLOL', 'LACTOL', 'NAPTHA', 'LINE', 'OTHER'};
	  elseif j==11
		val={'WoodHoe70', 'Motter70', 'Albert70', 'Motter94'};
	  elseif j==12
		val={'821', '802', '813', '824', '815', '816', '827', '828'};
	  elseif j==13
		val={'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'};
	  elseif j==14
		val={'CATALOG', 'SPIEGEL', 'TABLOID'};
	  elseif j==15
		val={'NorthUS', 'SouthUS', 'CANADIAN', 'SCANDANAVIAN', 'mideuropean'};
	  elseif j==16
		val={'1910', '1911'};
	  else
		x(i_fich,i,j) = fscanf(f, '%g',1); fscanf(f,'%c',1);
%  		printf('%g ', x(i_fich,i,j))
		continue
	  end
	  t=fscanf(f,'%s',1); n=length(val); a=2/(n-1); b=(1+n)/(1-n);
	  for k=1:n
		if strcmp(t,val{k})
		  x(i_fich,i,j)=a*k+b; break
		end
	  end
%  	  printf('%s ', t)
	  fscanf(f,'%c',1);   % descarta o espazo
	end	
	t=fscanf(f,'%s',1);    	% lectura da clase
	if strcmp(t,'band')
	  cl(i_fich,i) = 0;
	elseif strcmp(t,'noband')
	  cl(i_fich,i) = 1;
	else
	  error('clase %s descoñecida', t)
	end
%  	printf('cl= %s\n', t)
%  	if i==2 exit end
  end
  fclose(f);
end
