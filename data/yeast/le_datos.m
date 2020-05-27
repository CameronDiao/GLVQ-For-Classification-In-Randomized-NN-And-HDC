printf('lendo problema %s ...\n', problema);

n_entradas= 8; n_clases= 10; n_fich= 1; fich{1}= 'yeast.data'; n_patrons(1)= 1484;

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
	fscanf(f,'%s',1);   %descarta o nome de secuencia
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%g',1);
	end	
	t = fscanf(f,'%s',1);  	% lectura da clase
	if strcmp(t, 'CYT')
	  cl(i_fich,i)=0;
	elseif strcmp(t, 'NUC')
	  cl(i_fich,i)=1;
	elseif strcmp(t, 'MIT')
	  cl(i_fich,i)=2;
	elseif strcmp(t, 'ME3')
	  cl(i_fich,i)=3;
	elseif strcmp(t, 'ME2')
	  cl(i_fich,i)=4;
	elseif strcmp(t, 'ME1')
	  cl(i_fich,i)=5;
	elseif strcmp(t, 'EXC')
	  cl(i_fich,i)=6;
	elseif strcmp(t, 'VAC')
	  cl(i_fich,i)=7;
	elseif strcmp(t, 'POX')
	  cl(i_fich,i)=8;
	elseif strcmp(t, 'ERL')
	  cl(i_fich,i)=9;
	else
	  error('clase %s descoñecida', t)
	end
  end
  fclose(f);
end
