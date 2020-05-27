printf('lendo problema %s ...\n', problema);

n_entradas= 16; n_clases= 2; n_fich= 1; fich{1}= 'house-votes-84.data'; n_patrons(1)= 435;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;
val={'n','y'}; n=length(val); a=2/(n-1); b=(1+n)/(1-n);
clase={'democrat', 'republican'};
for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	t=''; while 1     % lectura de clase
	  c = fscanf(f, '%c',1);
	  if c==',' break end
	  t=strcat(t,c); 
	end
	if strcmp(t, 'democrat')
	  cl(i_fich,i)=0;
	elseif strcmp(t, 'republican')
	  cl(i_fich,i)=1;
	else
	  error('clase %s descoñecida\n', t)
	end
	for j = 1:n_entradas
	  t = fscanf(f,'%c',1); fscanf(f,'%c',1);
	  if t ~= '?'
		if t=='n'
		  x(i_fich,i,j) = -1;
		elseif t=='y'
		  x(i_fich,i,j) = -1;
		else
		  error('t=%c descoñecido\n', t)
		end
	  else
		x(i_fich,i,j) = 0;
	  end
	end	
  end
  fclose(f);
end
