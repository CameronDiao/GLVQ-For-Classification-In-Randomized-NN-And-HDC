printf('lendo problema %s ...\n', problema);

n_entradas= 16; n_clases= 2; n_fich= 1; fich{1}= 'bank.csv'; n_patrons(1)= 4521; %fich{2}= ' '; n_patrons(2)= 0;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  fscanf(f,'%s',1);  % le e descarta a 1ª fila
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  if j==2
		val={'admin.','unknown','unemployed','management','housemaid','entrepreneur','student','blue-collar','self-employed', 'retired', 'technician', 'services'};
	  elseif j==3
		val={'married','divorced','single'};
	  elseif j==4
		val={ 'unknown','secondary','primary','tertiary'};
	  elseif (j==5 || j==7 || j==8)
		val={'yes','no'};
	  elseif j==9
		val={'unknown','telephone','cellular'};
	  elseif j==11
		val={'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'};
	  elseif j==16
		val={'unknown','other','failure','success'};
	  else  % j=1,6,10,12,13,14,15
		x(i_fich,i,j)=fscanf(f,'%g',1); fscanf(f,'%c',1);  % le e descarta o carácter ; de separación entre entradas 
		continue
	  end
	  t=''; fscanf(f,'%c',1); % le e descarta o carácter " anterior á cadea de caracteres
	  while 1
		c = fscanf(f, '%c',1);
		if c=='"'
		  break
		end
		t=strcat(t,c); 
	  end
	  n=length(val); a=2/(n-1); b=(1+n)/(1-n);
	  for k=1:n
		if strcmp(t,val{k})
		  x(i_fich,i,j)=a*k+b; break
		end
	  end
	  fscanf(f,'%c',1);  % le e descarta o carácter ; de separación entre entradas 
	end
	% lectura da clase
	t=''; fscanf(f,'%c',1); % le e descarta o carácter " anterior á cadea de caracteres da clase
	while 1
	  c = fscanf(f, '%c',1);
	  if c=='"'
		break
	  end
	  t=strcat(t,c); 
	end
	if strcmp(t, 'no')
	  cl(i_fich,i) = 0;
	elseif strcmp(t,'yes')
	  cl(i_fich,i) = 1;
	else
	  error('clase <%s> descoñecida!')
	end
  end
  fclose(f);
end
