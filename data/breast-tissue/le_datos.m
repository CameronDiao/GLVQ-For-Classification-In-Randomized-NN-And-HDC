printf('lendo problema %s ...\n', problema);

n_entradas= 9; n_clases= 6; n_fich= 1; fich{1}= 'BreastTissue.csv'; n_patrons(1)= 106;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;
clase={'car', 'fad', 'mas', 'gla', 'con', 'adi'};

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	fscanf(f,'%i',1); fscanf(f,'%c',1);   % le ID e coma
	t=''; while 1    % lectura de clase
	  c = fscanf(f, '%c',1);
	  if c==',' break end
	  t=strcat(t,c); 
	end
	for j=1:n_clases
	  if strcmp(t, clase{j})
		cl(i_fich,i)=j-1; break
	  end
	end
%  	printf('cl= %s: ', t)
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%g',1); fscanf(f, '%c',1);
%  	  printf('%f ', x(i_fich,i,j))
	end	
%  	printf('\n')
%  	if i==2 exit end
  end
  fclose(f);
end
