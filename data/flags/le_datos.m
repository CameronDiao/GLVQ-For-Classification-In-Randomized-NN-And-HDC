printf('lendo problema %s ...\n', problema);

n_entradas= 28; n_clases= 8; n_fich= 1; fich{1}= 'flag.data'; n_patrons(1)= 194;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;
val={'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'brown'}; n=length(val); a=2/(n-1); b=(1+n)/(1-n);

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	fscanf(f,'%s',1); k=1;
	for j = 1:5
	  x(i_fich,i,k) = fscanf(f,'%g',1); k=k+1;
	end	
	cl(i_fich,i)= fscanf(f,'%i',1);   	% lectura da clase
	for j=1:10
	  x(i_fich,i,k) = fscanf(f,'%g',1); k=k+1;
	end
	t=fscanf(f,'%s',1);  % le e codifica a cor
	for l=1:n
	  if strcmp(t,val{l})
		x(i_fich,i,k)=a*l+b; k=k+1; break
	  end
	end
	for j=1:10
	  x(i_fich,i,k) = fscanf(f,'%g',1); k=k+1;
	end
	t=fscanf(f,'%s',1);  % le e codifica a cor
	for l=1:n
	  if strcmp(t,val{l})
		x(i_fich,i,k)=a*l+b; k=k+1; break
	  end
	end
	t=fscanf(f,'%s',1);  % le e codifica a cor
	for l=1:n
	  if strcmp(t,val{l})
		x(i_fich,i,k)=a*l+b; k=k+1; break
	  end
	end
  end
  fclose(f);
end
