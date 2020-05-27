printf('lendo problema %s ...\n', problema);

n_entradas= 60; n_clases= 3; n_fich= 1; fich{1}= 'splice.data'; n_patrons(1)= 3190;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;
val={'A', 'C', 'G', 'T'}; n=length(val);

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	t= fscanf(f,'%s',1);
	if strcmp(t, 'EI')
	  cl(i_fich,i)=0;
	elseif strcmp(t, 'IE')
	  cl(i_fich,i)=1;
	elseif strcmp(t, 'N')
	  cl(i_fich,i)=2;
	else
	  error('clase %s descoñecida', t)
	end
	fscanf(f,'%s',1);   % le e descarta o ID
	t=fscanf(f,'%s',1);
	for j = 1:n_entradas
	  for k=1:n
		if t(j)==val{k}
		  x(i_fich,i,j)=k; break
		end
	  end
	end	
  end
  fclose(f);
end
