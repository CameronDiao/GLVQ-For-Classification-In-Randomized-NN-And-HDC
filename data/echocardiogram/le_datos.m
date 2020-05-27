printf('lendo problema %s ...\n', problema);

n_entradas= 10; n_clases= 2; n_fich= 1; fich{1}= 'echocardiogram.data'; n_patrons(1)= 131;

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
	t=fscanf(f,'%c',1); % le e descarta survival
	if t~='?'
	  fseek(f,-1,SEEK_CUR); fscanf(f, '%g',1);
	end
	fscanf(f,'%c',1);
	cl(i_fich,i)=fscanf(f,'%i',1); fscanf(f,'%c',1);
%  	printf('i= %i cl= %i: ', i, cl(i_fich,i))
	for j = 1:n_entradas
	  t = fscanf(f,'%c',1);
	  if t ~= '?'
		fseek(f,-1,SEEK_CUR); t= fscanf(f, '%g',1); x(i_fich,i,j) = t;
%  		printf('%g ', t)
	  else
%  		printf('? ')
		x(i_fich,i,j) = 0;
	  end
	  fscanf(f,'%c',1);
	  if j==8
		fscanf(f,'%s',1); fscanf(f,'%c',1);
	  end  % le e descarta name
	end
%  	printf('\n')
%    	if i==2 exit end
  end
  fclose(f);
end
