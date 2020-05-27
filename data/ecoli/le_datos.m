printf('lendo problema %s ...\n', problema);

n_entradas= 7; n_clases= 8; n_fich= 1; fich{1}= 'ecoli.data'; n_patrons(1)= 336;

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
	t=fscanf(f,'%s',1); % le e descarta Sequence Name
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%g',1);
	end
	t= fscanf(f,'%s',1);  % lectura de clase
	if strcmp(t, 'cp')
	  cl(i_fich,i)=0;
	elseif strcmp(t,'im') 
	  cl(i_fich,i)=1;
	elseif strcmp(t,'pp')
	  cl(i_fich,i)=2;
	elseif strcmp(t,'imU')
	  cl(i_fich,i)=3;
	elseif strcmp(t,'om')
	  cl(i_fich,i)=4;
	elseif strcmp(t,'omL')
	  cl(i_fich,i)=5;
	elseif strcmp(t,'imL')
	  cl(i_fich,i)=6;
	elseif strcmp(t,'imS')
	  cl(i_fich,i)=7;
	else
	  error('clase %s descoñecida', t)
	end
  end
  fclose(f);
end
