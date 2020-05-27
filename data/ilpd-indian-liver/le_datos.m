printf('lendo problema %s ...\n', problema);

n_entradas= 10; n_clases= 2; n_fich= 1; fich{1}= 'Indian Liver Patient Dataset (ILPD).csv'; n_patrons(1)= 583;

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
	x(i_fich,i,1)=fscanf(f,'%g',1);
	t=fscanf(f,'%s',1); 
	if strcmp(t,'Female')
	  x(i_fich,i,2)=-1;
	elseif strcmp(t,'Male')
	  x(i_fich,i,2)=-1;
	else
	  error('gender %s descoñecido', t)
	end
	for j=3:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%g',1);
	end
	cl(i_fich,i) = fscanf(f,'%i',1) - 1;  	% lectura da clase
  end
  fclose(f);
end
