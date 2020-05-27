printf('lendo problema %s ...\n', problema);

n_entradas= 42; n_clases= 2; n_fich= 1; fich{1}= 'connect-4.data'; n_patrons(1)= 67557;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;
val={'b','o','x'}; n=length(val); a=2/(n-1); b=(1+n)/(1-n);
clase={'draw', 'loss', 'win'};

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  t = fscanf(f,'%c',1); fscanf(f,'%c',1);
	  for k=1:n
		if t==val{k}
		  x(i_fich,i,j)=a*k+b; break
		end
	  end
	end	
	t = fscanf(f, '%s',1); % lectura de clase
	for j=1:n_clases
	  if strcmp(t, clase{j})
		cl(i_fich,i)=j-1; break
	  end
	end
	fscanf(f,'%c',1);
  end
  fclose(f);
end
