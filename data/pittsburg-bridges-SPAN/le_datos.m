printf('lendo problema %s ...\n', problema);

n_entradas= 7; n_clases= 3; n_fich= 1; fich{1}= 'bridges.data.version2'; n_patrons(1)= 108;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);
n_patrons_total = sum(n_patrons); n_iter=0;
n_patrons_invalidos=0;  % nº de patróns onde a clase é descoñecida (?)

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	fscanf(f,'%s',1);   % descarta ID
	for j = 1:n_entradas
	  t = fscanf(f,'%s',1);
	  if t=='?'
		x(i_fich,i,j)=0; continue
	  end
	  if j==1
		val={'A', 'M', 'O'};
	  elseif j==3
		val={'CRAFTS', 'EMERGING', 'MATURE', 'MODERN'};
	  elseif j==4
		val={'WALK', 'AQUEDUCT', 'RR', 'HIGHWAY'};
	  elseif j==5
		val={'SHORT', 'MEDIUM', 'LONG'};
	  elseif j==7
		val={'N', 'G'};
	  else
		x(i_fich,i,j) = str2double(t); continue
	  end
	  n=length(val);
	  for k=1:n
		if strcmp(t,val{k})
		  x(i_fich,i,j)=k; break
		end
	  end
	end	
	fscanf(f,'%s',2);
	t=fscanf(f,'%s',1);  % lectura de clase
	if strcmp(t, 'SHORT')
	  cl(i_fich,i)=0;
	elseif strcmp(t, 'MEDIUM')
	  cl(i_fich,i)=1;
	elseif strcmp(t, 'LONG')
	  cl(i_fich,i)=2;
	elseif strcmp(t, '?')
	  n_patrons_invalidos = n_patrons_invalidos + 1;
%  	  printf('patrón %i inválido por ter clase descoñecida\n', i)
	  fscanf(f,'%s',2);
	  continue
	elseif
	  error('clase %s descoñecida', t)
	end
%  	printf('cl= %s\n', t)
	fscanf(f,'%s',2);
%  	if i==2 exit end
  end
  fclose(f);
end

n_patrons = n_patrons - n_patrons_invalidos;
